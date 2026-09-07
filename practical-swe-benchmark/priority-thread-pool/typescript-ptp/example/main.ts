import { PriorityThreadPool } from "../src/priority-thread-pool.js";

const pool = new PriorityThreadPool(2, new URL("./tasks.js", import.meta.url));
try {
  await Promise.all([1, 10, 5].map(async (priority) => {
    const mesh = await pool.submit<Int32Array>(priority, "computeMesh", priority * 100);
    // Promise continuations execute on the parent JS event loop, not a worker.
    console.log(`Upload mesh to GPU: ${mesh.length} vertices`);
  }));
} finally {
  await pool.shutdown();
}
