use std::collections::VecDeque;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle, ThreadId};

type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Debug, PartialEq, Eq)]
pub enum SubmitError {
    InvalidPriority,
    Closed,
}

#[derive(Debug, PartialEq, Eq)]
pub enum TaskError {
    Panicked(String),
    WorkerDisconnected,
}

pub struct Task<T>(mpsc::Receiver<Result<T, TaskError>>);
impl<T> Task<T> {
    pub fn join(self) -> Result<T, TaskError> {
        self.0.recv().unwrap_or(Err(TaskError::WorkerDisconnected))
    }
}

struct State {
    queues: [VecDeque<Job>; 11],
    closed: bool,
}

struct Shared {
    state: Mutex<State>,
    ready: Condvar,
}

pub struct PriorityThreadPool {
    shared: Arc<Shared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    worker_ids: Vec<ThreadId>,
}

impl PriorityThreadPool {
    pub fn new(workers: usize) -> io::Result<Self> {
        if workers == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "workers must be positive",
            ));
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queues: std::array::from_fn(|_| VecDeque::new()),
                closed: false,
            }),
            ready: Condvar::new(),
        });
        let mut handles = Vec::with_capacity(workers);
        for i in 0..workers {
            let state = Arc::clone(&shared);
            match thread::Builder::new()
                .name(format!("priority-worker-{i}"))
                .spawn(move || run(state))
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    shared.state.lock().unwrap().closed = true;
                    shared.ready.notify_all();
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(error);
                }
            }
        }
        let worker_ids = handles.iter().map(|h| h.thread().id()).collect();
        Ok(Self {
            shared,
            workers: Mutex::new(handles),
            worker_ids,
        })
    }

    pub fn submit<F, T>(&self, priority: i32, task: F) -> Result<Task<T>, SubmitError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        if !(0..=10).contains(&priority) {
            return Err(SubmitError::InvalidPriority);
        }
        let (sender, receiver) = mpsc::channel();
        let job = Box::new(move || {
            let result = catch_unwind(AssertUnwindSafe(task)).map_err(|panic| {
                let message = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "non-string panic".to_string()
                };
                TaskError::Panicked(message)
            });
            let _ = sender.send(result); // dropping a result does not cancel work
        });
        let mut state = self.shared.state.lock().unwrap();
        if state.closed {
            return Err(SubmitError::Closed);
        }
        state.queues[priority as usize].push_back(job);
        self.shared.ready.notify_one();
        Ok(Task(receiver))
    }

    /// Reject new work and drain accepted jobs. Nonblocking; safe in a task.
    pub fn close(&self) {
        self.shared.state.lock().unwrap().closed = true;
        self.shared.ready.notify_all();
    }

    /// Close and join workers. Call from an owner thread, not a pool task.
    pub fn shutdown(&self) -> Result<(), &'static str> {
        if self.worker_ids.contains(&thread::current().id()) {
            return Err("a worker cannot join its own pool; use close()");
        }
        self.close();
        for worker in self.workers.lock().unwrap().drain(..) {
            let _ = worker.join();
        }
        Ok(())
    }
}

impl Drop for PriorityThreadPool {
    fn drop(&mut self) {
        self.close();
        // If the last Arc<Pool> is dropped inside a task, detach workers rather
        // than blocking that task. They retain Shared and still drain the queue.
        let on_worker = self.worker_ids.contains(&thread::current().id());
        for worker in self.workers.get_mut().unwrap().drain(..) {
            if !on_worker {
                let _ = worker.join();
            }
        }
    }
}

fn run(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut state = shared.state.lock().unwrap();
            loop {
                if let Some(job) = state.queues.iter_mut().rev().find_map(VecDeque::pop_front) {
                    break job;
                }
                if state.closed {
                    return;
                }
                state = shared.ready.wait(state).unwrap();
            }
        };
        job();
    }
}
