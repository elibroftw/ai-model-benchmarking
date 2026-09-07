namespace PriorityPools;

/// <summary>Fixed dedicated threads; highest queued priority first, FIFO ties.</summary>
public sealed class PriorityThreadPool : IDisposable
{
    private readonly object gate = new();
    private readonly Lock joinGate = new();
    private readonly Queue<Action>[] queues = [.. Enumerable.Range(0, 11).Select(_ => new Queue<Action>())];
    private readonly Thread[] workers;
    private int pending;
    private bool closed;

    public PriorityThreadPool(int workerCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
        workers = [.. Enumerable.Range(0, workerCount).Select(i => new Thread(Run)
        {
            Name = $"priority-worker-{i}",
            IsBackground = false
        })];
        int started = 0;
        try
        {
            foreach (var worker in workers) { worker.Start(); started++; }
        }
        catch
        {
            Close();
            for (int i = 0; i < started; i++) workers[i].Join();
            throw;
        }
    }

    // Tasks are synchronous computations. Do not pass async delegates: their
    // continuations would escape the fixed set of dedicated workers.
    public Task<T> Submit<T>(int priority, Func<T> computation)
    {
        if (priority < 0 || priority > 10)
            throw new ArgumentOutOfRangeException(nameof(priority));
        ArgumentNullException.ThrowIfNull(computation);
        var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (gate)
        {
            if (closed) throw new InvalidOperationException("pool is closed");
            queues[priority].Enqueue(() =>
            {
                try { completion.SetResult(computation()); }
                catch (Exception error) { completion.SetException(error); }
            });
            pending++;
            Monitor.Pulse(gate);
        }
        return completion.Task;
    }

    /// <summary>Reject new tasks and drain accepted work. Safe inside a task.</summary>
    public void Close()
    {
        lock (gate) { closed = true; Monitor.PulseAll(gate); }
    }

    /// <summary>Close and join; only call from outside this pool's workers.</summary>
    public void Shutdown()
    {
        if (workers.Contains(Thread.CurrentThread))
            throw new InvalidOperationException("a worker cannot join its pool; use Close()");
        Close();
        lock (joinGate)
            foreach (var worker in workers) worker.Join();
    }

    public void Dispose() => Shutdown();

    private void Run()
    {
        while (true)
        {
            Action? job = null;
            lock (gate)
            {
                while (pending == 0 && !closed) Monitor.Wait(gate);
                if (pending == 0) return;
                for (int priority = 10; priority >= 0; priority--)
                {
                    if (queues[priority].Count == 0) continue;
                    job = queues[priority].Dequeue();
                    pending--;
                    break;
                }
            }
            job!(); // user code runs outside the scheduling lock
        }
    }
}
