import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit.SECONDS
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.jupiter.api.Timeout

private fun <T> CompletableFuture<T>.result(): T = get(5, SECONDS)

@Timeout(15)
class PriorityThreadPoolTest {
    @Test
    fun priorityFifoErrorsAndGracefulDrain() {
        val pool = PriorityThreadPool(1)
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        try {
            val running = pool.submit(0) {
                started.countDown()
                check(release.await(5, SECONDS))
            }
            assertTrue(started.await(5, SECONDS))
            val order = mutableListOf<Int>()
            val tasks = listOf(0, 10, 5, 10, 0).mapIndexed { i, priority ->
                pool.submit(priority) { order.add(i); i }
            }
            val failed = pool.submit(7) { error("task failure") }
            pool.close()
            assertFailsWith<IllegalStateException> { pool.submit(0) { 0 } }
            release.countDown()
            pool.shutdown()
            pool.shutdown()
            running.result()
            assertEquals(listOf(1, 3, 2, 0, 4), order)
            assertEquals((0..4).toList(), tasks.map { it.result() })
            val failure = assertFailsWith<ExecutionException> { failed.result() }
            assertEquals("task failure", failure.cause?.message)
        } finally {
            release.countDown()
            pool.shutdown()
        }
    }

    @Test
    fun fixedParallelism() {
        val pool = PriorityThreadPool(3)
        val entered = CountDownLatch(3)
        val gate = CountDownLatch(1)
        try {
            val active = AtomicInteger()
            val maximum = AtomicInteger()
            val tasks = List(8) {
                pool.submit(0) {
                    val count = active.incrementAndGet()
                    maximum.accumulateAndGet(count) { a, b -> maxOf(a, b) }
                    entered.countDown()
                    try {
                        check(gate.await(5, SECONDS))
                        Thread.currentThread()
                    } finally {
                        active.decrementAndGet()
                    }
                }
            }
            assertTrue(entered.await(5, SECONDS))
            gate.countDown()
            assertEquals(3, tasks.map { it.result() }.toSet().size)
            assertEquals(3, maximum.get())
        } finally {
            gate.countDown()
            pool.shutdown()
        }
    }

    @Test
    fun validationAndEmptyShutdown() {
        for (workers in listOf(0, -1)) {
            assertFailsWith<IllegalArgumentException> { PriorityThreadPool(workers) }
        }
        val pool = PriorityThreadPool(2)
        try {
            for (priority in listOf(-1, 11)) {
                assertFailsWith<IllegalArgumentException> { pool.submit(priority) { 0 } }
            }
        } finally {
            pool.shutdown()
        }
        pool.shutdown()
        assertFailsWith<IllegalStateException> { pool.submit(10) { 0 } }
    }

    @Test
    fun workerCannotJoinItsOwnPoolButCanCloseIt() {
        val pool = PriorityThreadPool(1)
        try {
            val failure = assertFailsWith<ExecutionException> {
                pool.submit(0) { pool.shutdown() }.result()
            }
            assertTrue(failure.cause is IllegalStateException)
            assertEquals(42, pool.submit(0) { 42 }.result())
            pool.submit(0) { pool.close() }.result()
            assertFailsWith<IllegalStateException> { pool.submit(0) { 0 } }
        } finally {
            pool.shutdown()
        }
    }

    @Test
    fun concurrentSubmitAndClose() {
        val pool = PriorityThreadPool(3)
        val start = CountDownLatch(1)
        val seeded = CountDownLatch(4)
        val accepted = ConcurrentLinkedQueue<CompletableFuture<Int>>()
        val errors = ConcurrentLinkedQueue<Throwable>()
        val ran = AtomicInteger()
        val rejected = AtomicInteger()
        val producers = List(4) {
            thread {
                try {
                    accepted.add(pool.submit(0) { ran.incrementAndGet() })
                    seeded.countDown()
                    check(start.await(5, SECONDS))
                    repeat(100) { priority ->
                        try {
                            accepted.add(pool.submit(priority % 11) { ran.incrementAndGet() })
                        } catch (_: IllegalStateException) {
                            rejected.incrementAndGet()
                        }
                    }
                } catch (error: Throwable) {
                    errors.add(error)
                }
            }
        }
        try {
            assertTrue(seeded.await(5, SECONDS))
            start.countDown()
            pool.close() // races with the producers' next execute() calls
            for (producer in producers) {
                producer.join(5000)
                assertFalse(producer.isAlive)
            }
            pool.shutdown()
            assertTrue(errors.isEmpty(), errors.toString())
            assertEquals(404, accepted.size + rejected.get())
            assertEquals(accepted.size, accepted.map { it.result() }.toSet().size)
            assertEquals(accepted.size, ran.get())
            assertFailsWith<IllegalStateException> { pool.submit(10) { 0 } }
        } finally {
            start.countDown()
            pool.shutdown()
            producers.forEach { it.join(5000) }
        }
    }
}
