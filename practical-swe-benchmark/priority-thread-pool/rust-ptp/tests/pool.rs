use priority_thread_pool::{PriorityThreadPool, SubmitError, TaskError};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

#[test]
fn priority_fifo_errors_and_drain() {
    let pool = PriorityThreadPool::new(1).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let running = pool
        .submit(0, move || {
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(TIMEOUT).unwrap();
        })
        .unwrap();
    started_rx.recv_timeout(TIMEOUT).unwrap();
    let order = Arc::new(Mutex::new(Vec::new()));
    let tasks: Vec<_> = [0, 10, 5, 10, 0]
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let order = Arc::clone(&order);
            pool.submit(p, move || {
                order.lock().unwrap().push(i);
                i
            })
            .unwrap()
        })
        .collect();
    let failed = pool.submit(7, || panic!("task failure")).unwrap();
    pool.close();
    assert!(matches!(pool.submit(0, || 1), Err(SubmitError::Closed)));
    release_tx.send(()).unwrap();
    pool.shutdown().unwrap();
    pool.shutdown().unwrap();
    running.join().unwrap();
    assert_eq!(*order.lock().unwrap(), vec![1, 3, 2, 0, 4]);
    for (i, task) in tasks.into_iter().enumerate() {
        assert_eq!(task.join().unwrap(), i);
    }
    assert_eq!(
        failed.join(),
        Err(TaskError::Panicked("task failure".into()))
    );
}

#[test]
fn fixed_parallelism_and_validation() {
    assert!(PriorityThreadPool::new(0).is_err());
    let pool = Arc::new(PriorityThreadPool::new(3).unwrap());
    for p in [-1, 11] {
        assert!(matches!(
            pool.submit(p, || ()),
            Err(SubmitError::InvalidPriority)
        ));
    }
    let (tx, rx) = mpsc::channel();
    let mut releases = Vec::new();
    let mut tasks = Vec::new();
    for _ in 0..3 {
        let tx = tx.clone();
        let (release_tx, release_rx) = mpsc::channel();
        releases.push(release_tx);
        tasks.push(
            pool.submit(0, move || {
                tx.send(std::thread::current().id()).unwrap();
                release_rx.recv_timeout(TIMEOUT).unwrap();
            })
            .unwrap(),
        );
    }
    let ids: std::collections::HashSet<_> =
        (0..3).map(|_| rx.recv_timeout(TIMEOUT).unwrap()).collect();
    assert_eq!(ids.len(), 3);
    for release in releases {
        release.send(()).unwrap();
    }
    for task in tasks {
        task.join().unwrap();
    }
    let worker_pool = Arc::clone(&pool);
    assert!(pool
        .submit(0, move || worker_pool.shutdown())
        .unwrap()
        .join()
        .unwrap()
        .is_err());
    assert_eq!(pool.submit(0, || 42).unwrap().join(), Ok(42));
}
