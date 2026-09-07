export type Request = { name: string; input: unknown };
export type Response =
  | { kind: "result"; value: unknown }
  | { kind: "error"; name: string; message: string; stack?: string };
