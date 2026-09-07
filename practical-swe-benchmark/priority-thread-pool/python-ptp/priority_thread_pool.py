"""Fixed-size priority workers. Higher priorities run first; ties are FIFO."""
from collections import deque
from concurrent.futures import Future
from threading import Condition, Thread, current_thread
from typing import Callable, TypeVar

T = TypeVar("T")


class PriorityThreadPool:
    def __init__(self, workers: int):
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        self._condition = Condition()
        self._queues = [deque() for _ in range(11)]
        self._closed = False
        self._threads = []
        try:
            for i in range(workers):
                thread = Thread(target=self._run, name=f"priority-worker-{i}")
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.shutdown()
            raise

    def submit(self, priority: int, fn: Callable[..., T], /, *args, **kwargs) -> Future[T]:
        if type(priority) is not int or not 0 <= priority <= 10:
            raise ValueError("priority must be an integer in [0, 10]")
        if not callable(fn):
            raise TypeError("task must be callable")
        future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("pool is closed")
            self._queues[priority].append((future, fn, args, kwargs))
            self._condition.notify()
        return future

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or any(self._queues))
                job = next((q.popleft() for q in reversed(self._queues) if q), None)
                if job is None:  # closed and drained
                    return
            future, fn, args, kwargs = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)

    def close(self):
        """Stop accepting tasks; queued work still runs. Safe inside a task."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def shutdown(self, wait: bool = True):
        """Close and optionally join. Blocking shutdown is for owner threads only."""
        if wait and current_thread() in self._threads:
            raise RuntimeError("a worker cannot join its own pool; use close()")
        self.close()
        if wait:
            for thread in self._threads:
                thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
