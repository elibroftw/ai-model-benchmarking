# Rust priority thread pool

Requires Rust 1.70+ / Cargo. No dependencies.

```sh
cargo test
cargo run --example render
cargo clippy --all-targets -- -D warnings
```

```rust
use priority_thread_pool::PriorityThreadPool;

let pool = PriorityThreadPool::new(4).unwrap();
let result = pool.submit(10, || 12 * 12).unwrap();
assert_eq!(result.join().unwrap(), 144);
pool.shutdown().unwrap();
```

Tasks are `FnOnce() -> T + Send + 'static`; results must be `Send + 'static`.
`submit` returns `Result<Task<T>, SubmitError>`; consuming `Task::join()` returns
`Result<T, TaskError>`. A closure can itself return a `Result` for ordinary
application errors. Unwinding panics become `TaskError::Panicked` without killing
the worker; Rust's normal panic hook still runs. `panic = "abort"` cannot be
caught. Dropping a task handle does not cancel work.

`close()` requests a drain. `shutdown()` closes and joins, rejecting self-join
from a pool worker. Owner-thread `Drop` also drains and joins. In the unusual
case where a task drops the last `Arc<PriorityThreadPool>`, Drop closes and
detaches the workers to avoid a self-join; the workers retain shared queue state
and finish accepted work. Explicit owner shutdown is preferable when completion
must be guaranteed before the process exits.

See the [shared scheduling contract](../README.md).
