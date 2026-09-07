import { threadId } from "node:worker_threads";

export function block(buffer: SharedArrayBuffer): number {
  const state = new Int32Array(buffer);
  Atomics.add(state, 0, 1);
  if (Atomics.wait(state, 1, 0, 5000) === "timed-out") throw new Error("gate timed out");
  return threadId;
}
export function record({ buffer, value }: { buffer: SharedArrayBuffer; value: number }): number {
  const state = new Int32Array(buffer);
  const index = Atomics.add(state, 0, 1);
  Atomics.store(state, index + 1, value);
  return value;
}
export function fail(): never { throw new Error("task failure"); }
export function unprintableError(): never { throw Object.create(null); }
export function uncloneable(): () => void { return () => {}; }
export function echo(value: unknown): unknown { return value; }
export async function asyncEcho(value: unknown): Promise<unknown> { return value; }
export async function asyncFail(): Promise<never> { throw new Error("async task failure"); }
export function exit(): never { process.exit(0); }
