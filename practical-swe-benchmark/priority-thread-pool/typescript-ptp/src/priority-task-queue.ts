import { FixedQueue, queueOptionsSymbol, type TaskQueue } from "piscina";

type Task = Parameters<TaskQueue["push"]>[0];

/** Piscina owns dispatch; this queue only chooses the next task. */
export class PriorityTaskQueue implements TaskQueue {
  private readonly queues = Array.from({ length: 11 }, () => new FixedQueue());

  get size(): number {
    return this.queues.reduce((total, queue) => total + queue.size, 0);
  }

  push(task: Task): void { this.bucket(task).push(task); }

  // Piscina may dequeue a task, find every worker busy, and return it. Putting
  // it at the tail would reorder FIFO ties. Used with stricterFIFO: true.
  unshift(task: Task): void { this.bucket(task).unshift(task); }

  shift(): Task | null {
    for (let priority = 10; priority >= 0; priority--) {
      const task = this.queues[priority].shift();
      if (task) return task;
    }
    return null;
  }

  remove(task: Task): void { this.bucket(task).remove(task); }

  private bucket(task: Task): FixedQueue {
    const { priority } = task[queueOptionsSymbol] as { priority: number };
    return this.queues[priority]; // validated by PriorityThreadPool.submit
  }
}
