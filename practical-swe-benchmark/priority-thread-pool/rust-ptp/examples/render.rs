use priority_thread_pool::PriorityThreadPool;

fn main() {
    let pool = PriorityThreadPool::new(2).unwrap();
    let meshes: Vec<_> = [1, 10, 5]
        .into_iter()
        .map(|priority| {
            pool.submit(priority, move || {
                (0..priority * 100).map(|i| i * i).collect::<Vec<_>>()
            })
            .unwrap()
        })
        .collect();
    for mesh in meshes {
        // Upload only on this caller/render thread.
        println!(
            "Upload mesh to GPU: {} vertices",
            mesh.join().unwrap().len()
        );
    }
}
