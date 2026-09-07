#include "priority_thread_pool.hpp"
#include <iostream>

int main() {
    PriorityThreadPool pool(2);
    std::vector<std::future<std::vector<int>>> meshes;
    for (int priority : {1, 10, 5}) {
        meshes.push_back(pool.submit(priority, [priority] {
            std::vector<int> vertices(100 * priority);
            for (std::size_t i = 0; i < vertices.size(); ++i) vertices[i] = i * i;
            return vertices;
        }));
    }
    for (auto& mesh : meshes) {
        // Only the calling/render thread touches the GPU.
        std::cout << "Upload mesh to GPU: " << mesh.get().size() << " vertices\n";
    }
}
