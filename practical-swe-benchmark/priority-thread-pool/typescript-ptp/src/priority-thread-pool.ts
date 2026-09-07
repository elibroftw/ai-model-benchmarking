import { Piscina, queueOptionsSymbol } from "piscina";
import { PriorityTaskQueue } from "./priority-task-queue.js";
import type { Request, Response } from "./protocol.js";

type ScheduledRequest = Request & { [queueOptionsSymbol]: { priority: number } };

/** Fixed Piscina workers with a highest-priority-first, FIFO-ties task queue. */
export class PriorityThreadPool {
  private readonly executor: Piscina<ScheduledRequest, Response>;
  private readonly pending = new Set<Promise<unknown>>();
  private closed = false;
  private failure?: Error;
  private destruction?: Promise<void>;
  private completion?: Promise<void>;

  constructor(workerCount: number, taskModule: URL) {
    if (!Number.isSafeInteger(workerCount) || workerCount <= 0) {
      throw new RangeError("workers must be a positive integer");
    }
    if (!(taskModule instanceof URL) || taskModule.protocol !== "file:") {
      throw new TypeError("taskModule must be a file URL to a compiled task module");
    }
    this.executor = new Piscina({
      filename: new URL("./worker.js", import.meta.url).href,
      workerData: { moduleUrl: taskModule.href },
      minThreads: workerCount,
      maxThreads: workerCount,
      concurrentTasksPerWorker: 1,
      taskQueue: new PriorityTaskQueue(),
      stricterFIFO: true,
    });
    this.executor.on("error", (error: Error) => this.fail(error));
  }

  /** Input is snapshotted; T must match the selected task export's result. */
  submit<T>(priority: number, name: string, input: unknown): Promise<T> {
    if (!Number.isInteger(priority) || priority < 0 || priority > 10) {
      throw new RangeError("priority must be an integer in [0, 10]");
    }
    if (typeof name !== "string" || !name) throw new TypeError("task export name is required");
    if (this.closed) throw new Error("pool is closed");
    const snapshot: unknown = structuredClone(input);
    // Cloning can invoke input getters, which may reenter close()/shutdown().
    if (this.closed) throw new Error("pool is closed");
    const result = this.executor.run({
      name, input: snapshot, [queueOptionsSymbol]: { priority },
    }).then((response) => {
      if (this.failure) throw this.failure;
      if (response.kind === "result") return response.value as T;
      const error = new Error(response.message);
      error.name = response.name;
      error.stack = response.stack;
      throw error;
    }, (error: Error) => {
      // Task errors are responses from the adapter. A rejected Piscina run is
      // an infrastructure failure; preserve fail-all rather than retry work.
      this.fail(error);
      throw this.failure;
    });
    this.pending.add(result);
    void result.then(() => this.pending.delete(result), () => this.pending.delete(result));
    return result;
  }

  /** Reject new tasks and drain accepted work without blocking the event loop. */
  close(): void {
    if (this.completion) return;
    this.closed = true;
    this.completion = (async () => {
      // Piscina's close has a default 30-second timeout. Wait for accepted jobs
      // first so graceful shutdown never aborts a long-running computation.
      await Promise.allSettled([...this.pending]);
      if (!this.failure) await this.executor.close();
      if (this.failure) {
        await this.destruction;
        throw this.failure;
      }
    })();
    // close() is intentionally void; shutdown() exposes any eventual failure.
    void this.completion.catch(() => {});
  }

  /** Wait for drain and worker exit, or reject on a worker infrastructure error. */
  shutdown(): Promise<void> {
    this.close();
    return this.completion!;
  }

  private fail(error: Error): void {
    if (this.failure) return;
    this.failure = error;
    this.destruction = this.executor.destroy();
    this.close();
  }
}
