export function computeMesh(size: number): Int32Array {
  return Int32Array.from({ length: size }, (_, i) => i * i);
}
