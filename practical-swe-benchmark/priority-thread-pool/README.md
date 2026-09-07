# Priority thread pools

Seven standalone implementations of the same fixed-worker priority scheduler.
Each directory includes a reusable pool, deterministic tests, and a small
compute-then-render example. No GPU or GUI framework is required.

| Language | Directory | Execution / result mechanism |
| --- | --- | --- |
| Rust | [rust-ptp](rust-ptp/) | OS threads, typed `Task<T>` |
| C# | [csharp-ptp](csharp-ptp/) | Dedicated `Thread`s, `Task<T>` |
| TypeScript | [typescript-ptp](typescript-ptp/) | Piscina worker threads, `Promise<T>` |
| Python | [python-ptp](python-ptp/) | `threading.Thread`, `concurrent.futures.Future` |
| C++ | [cpp-ptp](cpp-ptp/) | `std::thread`, `std::future<T>` |
| Kotlin | [kotlin-ptp](kotlin-ptp/) | JVM `ThreadPoolExecutor`, `CompletableFuture<T>` |
| Go | [go-ptp](go-ptp/) | Fixed worker goroutines, channel-backed `Future` |

## Shared contract

- The worker count is fixed and must be positive. More tasks than workers may
  be submitted, including while other tasks are running.
- Priorities are **integers from 0 through 10**, inclusive; **10 is highest**.
  Invalid priorities are rejected, not clamped.
- Every dequeue selects the highest-priority **currently queued** task. Equal
  priorities are FIFO in enqueue order (lock acquisition order for concurrent
  producers).
- Scheduling is **non-preemptive**. A high-priority submission cannot interrupt
  work already assigned to a worker. With multiple workers, dispatch order is
  not necessarily observable start order, and completion order is not guaranteed.
- Every accepted task runs once, and its result or ordinary task failure is
  delivered through its future. Python additionally supports cancelling work
  that has not started through its standard future API.
- User computations execute outside scheduling locks. Idle workers sleep on
  conditions, Atomics waits, or message ports rather than busy-spin.
- `close()` / `Close()` is nonblocking: it rejects new submissions and lets
  accepted work drain. `shutdown()` / `Shutdown()` also waits for workers to
  exit (an asynchronous promise in TypeScript). Both are idempotent.
- A submit racing with close is either accepted and drained or rejected; it is
  never silently lost. Blocking shutdown must be called from an owner thread,
  **not from a task in that pool**. Tasks may request nonblocking close.
- These are unbounded queues with strict priority, not real-time schedulers.
  Sustained high-priority traffic can starve low priorities. There is no aging,
  reprioritization, forced task cancellation, deadline, or shutdown timeout.
  Shutdown requires computations to eventually return.

Most implementations use **11 FIFO buckets** because the priority domain has
only 11 values: enqueue is O(1), dequeue scans at most 11 buckets, and queued
storage is O(number of tasks). TypeScript supplies this scheduling policy to
**Piscina**, using its `FixedQueue` buckets while delegating worker management,
message transport, and termination to the library.

Kotlin instead uses the standard **`ThreadPoolExecutor` with a
`PriorityBlockingQueue`**. Its heap orders comparable jobs by priority and then
an enqueue sequence for FIFO ties, with O(log queued tasks) heap operations.
The JDK handles queue synchronization and worker lifecycle. The remaining
native-thread implementations use one lock for queue and lifecycle state.

## Keeping rendering single-threaded

Workers should compute CPU-side data only. The owner/render thread consumes
future results and alone calls the UI/GPU API. Future completion callbacks are
**not universally render-thread callbacks**: Python and Kotlin callbacks may run
on the worker, and C# needs the GUI's `SynchronizationContext` or dispatcher.
Dispatch explicitly to your framework's UI/render queue when necessary.

The examples are small command-line demonstrations that wait for results and
print a simulated GPU upload on the caller thread. Do not copy their blocking
wait loops into a live UI frame: poll ready results each frame, enqueue completed
data in a thread-safe render queue, or await using the framework's UI dispatcher.
TypeScript's example uses promise continuations on the parent event loop; Go's
example pins the rendering goroutine to its OS thread.

Avoid another common pool deadlock: all workers must not synchronously wait for
child jobs queued to the same saturated pool. Tasks also must not wait for UI
work while the UI is synchronously joining the pool.

## Run tests

See each directory's README for prerequisites, commands, and API examples.
From this directory, with all seven toolchains installed:

```sh
bash test-all.sh
# Or select languages:
bash test-all.sh python cpp typescript
```

The script installs the locked TypeScript dependencies (including Piscina).
Kotlin uses its Gradle wrapper, which downloads Gradle, the Kotlin compiler, and
test dependencies; only a compatible JDK needs to be installed. Other language
toolchains must already be available. Tests use gates/latches to occupy workers before enqueueing
a priority batch, rather than timing-dependent sleeps. They exercise priority
ordering, FIFO ties, actual concurrent workers, results, failures, invalid inputs,
shutdown draining, and rejection after close. TypeScript additionally tests
worker exit and module-load failures; Go can be tested with the race detector.

### Runtime limitations

- On normal GIL-enabled CPython, threads do not speed up pure-Python CPU-bound
  computation. Use GIL-releasing native code, I/O workloads, or free-threaded
  CPython; a process pool would be a different implementation.
- Go schedules worker goroutines onto OS threads. `GOMAXPROCS` controls CPU
  parallelism; the pool bounds concurrent jobs, not OS-thread count.
- TypeScript tasks are named module exports, not transferred closures. Inputs
  and outputs must support structured cloning. Shared buffers require Atomics
  or other appropriate synchronization.
- Task exceptions/panics are contained, but process termination, fatal runtime
  errors, or forcibly killing native workers are not recoverable task errors.
  TypeScript treats infrastructure errors reported by Piscina as pool failures,
  rejecting outstanding work rather than silently retrying side effects.
