import threading
import unittest

from priority_thread_pool import PriorityThreadPool


class PoolTests(unittest.TestCase):
    def test_priority_fifo_errors_and_drain(self):
        started, release = threading.Event(), threading.Event()
        order = []
        pool = PriorityThreadPool(1)
        def blocker():
            started.set()
            self.assertTrue(release.wait(5))
        running = pool.submit(0, blocker)
        try:
            self.assertTrue(started.wait(5))
            futures = [pool.submit(p, lambda value=i: order.append(value) or value)
                       for i, p in enumerate([0, 10, 5, 10, 0])]
            failed = pool.submit(7, lambda: 1 / 0)
            cancelled = pool.submit(9, lambda: self.fail("cancelled task ran"))
            self.assertTrue(cancelled.cancel())
            pool.close()
            with self.assertRaises(RuntimeError):
                pool.submit(1, lambda: None)
        finally:
            release.set()
            pool.shutdown()
        running.result(5)
        self.assertEqual([f.result(5) for f in futures], list(range(5)))
        self.assertEqual(order, [1, 3, 2, 0, 4])
        with self.assertRaises(ZeroDivisionError):
            failed.result(5)
        pool.shutdown()  # idempotent

    def test_fixed_parallelism(self):
        with PriorityThreadPool(3) as pool:
            gate = threading.Barrier(4)
            futures = [pool.submit(0, lambda: (gate.wait(5), threading.get_ident())[1])
                       for _ in range(3)]
            gate.wait(5)  # all three workers must be executing simultaneously
            self.assertEqual(len({f.result(5) for f in futures}), 3)

    def test_validation_and_worker_shutdown(self):
        for workers in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                PriorityThreadPool(workers)
        with PriorityThreadPool(1) as pool:
            for priority in [-1, 11, True, 0.5]:
                with self.assertRaises(ValueError):
                    pool.submit(priority, lambda: None)
            with self.assertRaises(TypeError):
                pool.submit(1, None)
            with self.assertRaises(RuntimeError):
                pool.submit(1, pool.shutdown).result(5)
            self.assertEqual(pool.submit(10, lambda: 42).result(5), 42)


if __name__ == "__main__":
    unittest.main()
