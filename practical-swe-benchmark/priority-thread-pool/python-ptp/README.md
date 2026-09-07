# Python priority thread pool

Requires Python 3.10+; standard library only.

```sh
python3 -m unittest -v
python3 example.py
```

```python
from priority_thread_pool import PriorityThreadPool

with PriorityThreadPool(4) as pool:
    result = pool.submit(10, lambda x: x * x, 12)
    print(result.result())  # 144; exceptions are re-raised here
```

`submit(priority, callable, *args, **kwargs)` returns a standard
`concurrent.futures.Future`. `future.cancel()` can prevent queued work from
running; it cannot interrupt a running task. `close()` or `shutdown(wait=False)`
requests a drain without joining. `shutdown()` and context-manager exit join all
workers. Blocking shutdown from a pool worker raises `RuntimeError`.

Workers are non-daemon: always close the pool or use `with`. Callbacks registered
with `add_done_callback` may execute on a worker, so dispatch UI work explicitly.
Arguments and returned objects are shared in-process; synchronize mutable data.
The GIL normally prevents parallel execution of pure-Python CPU computations;
threads still help for I/O and native operations that release it, and can run
CPU code in parallel on free-threaded Python builds.

See the [shared scheduling contract](../README.md).
