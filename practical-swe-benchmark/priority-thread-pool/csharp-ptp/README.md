# C# priority thread pool

Requires .NET 8 SDK. No NuGet dependencies. `PriorityThreadPool.cs` is reusable;
the console project runs self-checking tests by default.

```sh
dotnet run
dotnet run -- --example
```

```csharp
using PriorityPools;

using var pool = new PriorityThreadPool(4);
Task<int> result = pool.Submit(10, () => 12 * 12);
int value = await result; // 144; task exceptions propagate through await
```

This uses dedicated `Thread`s, not the global .NET thread pool. Each task is a
**synchronous** `Func<T>` computation; passing an async delegate would return a
nested task and move continuations outside the fixed workers. For a void-like
job, return a sentinel value. `TaskCompletionSource` continuations are scheduled
asynchronously so user continuations do not run inline on the worker.

`Close()` requests a drain; `Shutdown()` and `Dispose()` drain and join.
Blocking shutdown from a worker throws `InvalidOperationException`. Workers are
foreground threads, so always dispose the pool. Cancellation is not provided.
In a GUI, `await` from its synchronization context or dispatch explicitly back
to the UI thread. Do not block the UI waiting for a task that needs UI work.

See the [shared scheduling contract](../README.md).
