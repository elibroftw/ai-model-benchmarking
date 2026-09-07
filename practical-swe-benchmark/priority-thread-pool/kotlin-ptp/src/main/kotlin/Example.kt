fun main() {
    val pool = PriorityThreadPool(2)
    try {
        val meshes = listOf(1, 10, 5).map { priority ->
            pool.submit(priority) { IntArray(priority * 100) { it * it } }
        }
        for (mesh in meshes) {
            // This demo uploads on the calling/render thread. A real UI should
            // poll completed futures or dispatch through its UI event loop.
            println("Upload mesh to GPU: ${mesh.get().size} vertices")
        }
    } finally {
        pool.shutdown()
    }
}
