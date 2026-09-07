# TypeScript priority thread pool

Requires Node.js 22+ and npm. Uses [Piscina](https://github.com/piscinajs/piscina)
with real `worker_threads`, not promise-only "parallelism". Piscina is a pinned
runtime dependency; TypeScript and Node types are development dependencies.

```sh
npm ci
npm test
npm run example
```

Piscina owns worker creation, message transport, dispatch, and termination.
`minThreads` and `maxThreads` are both the requested worker count, with one task
per worker. The custom `PriorityTaskQueue` uses 11 of Piscina's `FixedQueue`
buckets and its `queueOptionsSymbol` metadata to select the highest priority.
`stricterFIFO` and the queue's `unshift` hook preserve FIFO ties when Piscina
returns an undispatched task to the queue.

Tasks are named exports in a **compiled JavaScript module**. For example:

```ts
// tasks.ts
export function square(value: number): number { return value * value; }
```

```ts
// main.ts (compiled alongside tasks.ts)
import { PriorityThreadPool } from "./src/priority-thread-pool.js";

const pool = new PriorityThreadPool(4, new URL("./tasks.js", import.meta.url));
try {
  const value = await pool.submit<number>(10, "square", 12); // 144
  // This continuation runs on the parent event loop; render here.
} finally {
  await pool.shutdown();
}
```

`submit<T>(priority, exportName, input)` returns `Promise<T>`. The caller is
responsible for matching `T` to the export's result; workers are a runtime
message boundary. Validation, closed-pool submission, and input clone errors
throw synchronously; computation errors reject the returned promise.

Inputs are structured-cloned at submission (ordinary objects are snapshots)
and sent to workers. Results must also be cloneable; non-cloneable results reject
the task without losing the worker. A small Piscina handler adapter dispatches
the chosen export, checks result cloneability, and returns task-error envelopes
so ordinary failures remain distinct from worker infrastructure errors. This
adds a result clone before Piscina's own message serialization. Closures cannot
be transferred. SharedArrayBuffer contents remain shared and require synchronization. Transfer lists and task
cancellation are not provided. Task modules load independently in each worker,
so module-local state is not shared.

Sync and async exports are supported, with at most one export invocation in
flight per worker. Do not spawn detached work or manipulate the pool's worker
message port from task code. Promise handlers execute on the parent event loop,
never on the computation worker.

`close()` drains asynchronously; `await shutdown()` also waits for worker exit.
The wrapper waits for accepted tasks to settle before invoking Piscina's close,
so Piscina's default 30-second close timeout cannot cut off long-running work.

A startup error or unexpected exit during a task fails the whole pool: the
wrapper calls Piscina's `destroy()` and rejects outstanding jobs plus shutdown.
It does not continue queued work on replacement workers or retry failed tasks,
because they may already have produced side effects. Ordinary task errors are
isolated. Always handle task rejections and shut down the pool to release its
threads.

See the [shared scheduling contract](../README.md).
