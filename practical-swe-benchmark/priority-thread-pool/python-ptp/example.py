from priority_thread_pool import PriorityThreadPool


def compute_mesh(size):
    return [i * i for i in range(size)]


if __name__ == "__main__":
    with PriorityThreadPool(2) as pool:
        meshes = [pool.submit(priority, compute_mesh, size)
                  for priority, size in [(1, 100), (10, 200), (5, 150)]]
        for mesh in meshes:
            # This runs on the caller/render thread, never on a pool worker.
            print("Upload mesh to GPU:", len(mesh.result()), "vertices")
