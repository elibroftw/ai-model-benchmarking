using PriorityPools;

static class Program
{
    static void Check(bool condition)
    {
        if (!condition) throw new Exception("test failed");
    }

    static void Throws<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T) { return; }
        throw new Exception($"expected {typeof(T).Name}");
    }

    static T Result<T>(Task<T> task) => task.WaitAsync(TimeSpan.FromSeconds(5)).GetAwaiter().GetResult();

    static void Test()
    {
        Throws<ArgumentOutOfRangeException>(() => new PriorityThreadPool(0));
        using var pool = new PriorityThreadPool(1);
        foreach (int priority in new[] { -1, 11 })
            Throws<ArgumentOutOfRangeException>(() => pool.Submit(priority, () => 0));
        Throws<ArgumentNullException>(() => pool.Submit<int>(0, null!));
        using var started = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var running = pool.Submit(0, () => { started.Set(); Check(release.Wait(5000)); return 0; });
        var order = new List<int>();
        var futures = new List<Task<int>>();
        Task<int> failed;
        try
        {
            Check(started.Wait(5000));
            int[] priorities = { 0, 10, 5, 10, 0 };
            for (int i = 0; i < priorities.Length; i++)
            {
                int value = i;
                futures.Add(pool.Submit(priorities[i], () => { order.Add(value); return value; }));
            }
            failed = pool.Submit<int>(7, () => throw new ApplicationException("task failure"));
            pool.Close();
            Throws<InvalidOperationException>(() => pool.Submit(0, () => 0));
        }
        finally { release.Set(); }
        pool.Shutdown();
        pool.Shutdown();
        Result(running);
        Check(order.SequenceEqual(new[] { 1, 3, 2, 0, 4 }));
        for (int i = 0; i < futures.Count; i++) Check(Result(futures[i]) == i);
        Throws<ApplicationException>(() => Result(failed));

        using var parallel = new PriorityThreadPool(3);
        using var entered = new CountdownEvent(3);
        using var gate = new ManualResetEventSlim();
        var tasks = Enumerable.Range(0, 3).Select(_ => parallel.Submit(0, () =>
        {
            entered.Signal();
            Check(gate.Wait(5000));
            return Environment.CurrentManagedThreadId;
        })).ToArray();
        try { Check(entered.Wait(5000)); }
        finally { gate.Set(); }
        Check(tasks.Select(Result).Distinct().Count() == 3);
        Throws<InvalidOperationException>(() => Result(parallel.Submit(0, () => { parallel.Shutdown(); return 0; })));
        Check(Result(parallel.Submit(0, () => 42)) == 42);
        Console.WriteLine("All tests passed");
    }

    static void Example()
    {
        using var pool = new PriorityThreadPool(2);
        var meshes = new[] { 1, 10, 5 }.Select(priority => pool.Submit(priority,
            () => Enumerable.Range(0, priority * 100).Select(i => i * i).ToArray())).ToArray();
        foreach (var mesh in meshes)
        {
            // This console demo stays on the calling/render thread. In a GUI,
            // await from its SynchronizationContext instead of blocking the UI.
            Console.WriteLine($"Upload mesh to GPU: {mesh.GetAwaiter().GetResult().Length} vertices");
        }
    }

    static void Main(string[] args)
    {
        if (args.Contains("--example")) Example(); else Test();
    }
}
