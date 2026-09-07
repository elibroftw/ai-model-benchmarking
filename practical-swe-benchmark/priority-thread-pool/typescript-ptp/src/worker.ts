import { workerData } from "piscina";
import type { Request, Response } from "./protocol.js";

const tasks = await import(workerData.moduleUrl);

/** Piscina handles threads, messaging, and completion. This adapter preserves
 * named exports and separates ordinary task errors from worker failures.
 */
export default async function execute({ name, input }: Request): Promise<Response> {
  try {
    if (!Object.hasOwn(tasks, name) || typeof tasks[name] !== "function") {
      throw new TypeError(`No task export named ${name}`);
    }
    const value: unknown = await tasks[name](input);
    // Detect clone failures here as task errors, before Piscina posts a result.
    return { kind: "result", value: structuredClone(value) };
  } catch (error) {
    let response: Response = { kind: "error", name: "Error", message: "Task threw an unprintable value" };
    try {
      const e = error instanceof Error ? error : new Error(String(error));
      response = {
        kind: "error", name: String(e.name), message: String(e.message),
        stack: e.stack === undefined ? undefined : String(e.stack),
      };
    } catch { /* Error getters/toString may themselves throw. Keep the fallback. */ }
    return response;
  }
}
