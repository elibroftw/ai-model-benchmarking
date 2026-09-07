import assert from "node:assert/strict";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { PriorityThreadPool } from "../src/priority-thread-pool.js";

const moduleUrl = new URL("./tasks.js", import.meta.url);
async function until(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 5000;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, "worker did not reach gate");
    await delay(1);
  }
}
function release(state: Int32Array): void {
  Atomics.store(state, 1, 1);
  Atomics.notify(state, 1);
}

test("priority, FIFO ties, task errors, input snapshot, and drain", async () => {
  const pool = new PriorityThreadPool(1, moduleUrl);
  const gate = new Int32Array(new SharedArrayBuffer(8));
  const running = pool.submit<number>(0, "block", gate.buffer);
  try {
    await until(() => Atomics.load(gate, 0) === 1);
    const order = new Int32Array(new SharedArrayBuffer(24));
    const tasks = [0, 10, 5, 10, 0].map((p, value) =>
      pool.submit<number>(p, "record", { buffer: order.buffer, value }));
    const failed = assert.rejects(pool.submit(7, "fail", null), /task failure/);
    const missing = assert.rejects(pool.submit(7, "missing", null), /No task export/);
    const cloneFailure = assert.rejects(pool.submit(7, "uncloneable", null));
    const oddError = assert.rejects(pool.submit(7, "unprintableError", null), /unprintable value/);
    const input = { value: 42 };
    const snapshot = pool.submit(1, "echo", input);
    input.value = -1;
    pool.close();
    assert.throws(() => pool.submit(0, "echo", null), /closed/);
    release(gate);
    await pool.shutdown();
    assert.ok(await running > 0);
    assert.deepEqual(await Promise.all(tasks), [0, 1, 2, 3, 4]);
    assert.deepEqual([...order].slice(1), [1, 3, 2, 0, 4]);
    assert.deepEqual(await snapshot, { value: 42 });
    await Promise.all([failed, missing, cloneFailure, oddError]);
    await pool.shutdown();
  } finally {
    release(gate);
    await pool.shutdown();
  }
});

test("fixed real parallel workers and validation", async () => {
  for (const n of [0, -1, 1.5, NaN]) assert.throws(() => new PriorityThreadPool(n, moduleUrl));
  const pool = new PriorityThreadPool(3, moduleUrl);
  const gate = new Int32Array(new SharedArrayBuffer(8));
  try {
    for (const p of [-1, 11, 0.5, NaN]) assert.throws(() => pool.submit(p, "echo", 0));
    assert.throws(() => pool.submit(0, "echo", () => {}));
    const tasks = Array.from({ length: 8 }, () => pool.submit<number>(0, "block", gate.buffer));
    await until(() => Atomics.load(gate, 0) === 3);
    release(gate);
    assert.equal(new Set(await Promise.all(tasks)).size, 3);
  } finally {
    release(gate);
    await pool.shutdown();
  }
});

test("worker exit rejects active and queued tasks instead of hanging", async () => {
  const pool = new PriorityThreadPool(1, moduleUrl);
  const results = Promise.allSettled([
    pool.submit(10, "exit", null), pool.submit(0, "echo", 42),
  ]);
  await assert.rejects(pool.shutdown(), /exited/);
  assert.ok((await results).every((r) => r.status === "rejected"));
});

test("async exports and primitive inputs retain their API", async () => {
  const pool = new PriorityThreadPool(1, moduleUrl);
  try {
    assert.equal(await pool.submit(10, "asyncEcho", 42), 42);
    await assert.rejects(pool.submit(0, "asyncFail", null), /async task failure/);
    assert.equal(await pool.submit(0, "echo", null), null);
    assert.equal(await pool.submit(0, "echo", undefined), undefined);
  } finally {
    await pool.shutdown();
  }
});

test("input getter closing the pool cannot enqueue work after shutdown", async () => {
  const pool = new PriorityThreadPool(1, moduleUrl);
  const input = { get value() { pool.close(); return 42; } };
  assert.throws(() => pool.submit(0, "echo", input), /closed/);
  await pool.shutdown();
});

test("empty shutdown and module loading failure", async () => {
  await new PriorityThreadPool(2, moduleUrl).shutdown();
  const pool = new PriorityThreadPool(1, new URL("./missing.js", import.meta.url));
  const task = assert.rejects(pool.submit(0, "echo", 0));
  await assert.rejects(pool.shutdown());
  await task;
});
