import assert from "node:assert/strict";
import { test } from "node:test";
import { queueOptionsSymbol } from "piscina";
import { PriorityTaskQueue } from "../src/priority-task-queue.js";

const task = (priority: number, id: number) => ({ [queueOptionsSymbol]: { priority }, id });

test("Piscina queue selects priorities with stable FIFO ties", () => {
  const queue = new PriorityTaskQueue();
  const tasks = [0, 10, 5, 10, 0].map(task);
  for (const task of tasks) queue.push(task);
  assert.equal(queue.size, 5);
  assert.deepEqual(Array.from({ length: 5 }, () => queue.shift()),
    [tasks[1], tasks[3], tasks[2], tasks[0], tasks[4]]);
  assert.equal(queue.size, 0);
  assert.equal(queue.shift(), null);
});

test("undispatched tasks return to the front of their priority, not ahead of higher priorities", () => {
  const queue = new PriorityTaskQueue();
  const first = task(5, 1), second = task(5, 2), high = task(10, 3);
  queue.push(first);
  queue.push(second);
  assert.equal(queue.shift(), first);
  queue.push(high);
  queue.unshift(first); // Piscina could not find an available worker
  assert.equal(queue.size, 3);
  assert.equal(queue.shift(), high);
  assert.equal(queue.shift(), first);
  assert.equal(queue.shift(), second);
});

test("queue removal and FIFO survive FixedQueue buffer rollover", () => {
  const queue = new PriorityTaskQueue();
  const tasks = Array.from({ length: 5000 }, (_, id) => task(7, id));
  for (const task of tasks) queue.push(task);
  const removed = tasks[2500];
  queue.remove(removed);
  queue.remove(removed); // idempotent
  assert.equal(queue.size, tasks.length - 1);
  for (const task of tasks) {
    if (task !== removed) assert.equal(queue.shift(), task);
  }
  assert.equal(queue.size, 0);
  assert.equal(queue.shift(), null);
  queue.push(task(0, 0));
  queue.remove(task(0, -1)); // unknown tasks do not remove other work
  assert.equal(queue.size, 1);
});
