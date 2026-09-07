import java.math.BigInteger
import java.util.concurrent.CompletableFuture
import java.util.concurrent.PriorityBlockingQueue
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadFactory
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** Standard JVM executor with highest-priority-first scheduling and FIFO ties. */
class PriorityThreadPool(workerCount: Int) {
    private val submissionLock = Any()
    private var sequence = BigInteger.ZERO
    private val onWorker = ThreadLocal.withInitial { false }
    private val executor: ThreadPoolExecutor

    init {
        require(workerCount > 0) { "workers must be positive" }
        val workerNumber = AtomicInteger()
        val factory = ThreadFactory { worker ->
            Thread({
                onWorker.set(true)
                try { worker.run() } finally { onWorker.remove() }
            }, "priority-worker-${workerNumber.getAndIncrement()}").apply { isDaemon = false }
        }
        executor = ThreadPoolExecutor(
            workerCount, workerCount, 0L, TimeUnit.MILLISECONDS,
            PriorityBlockingQueue<Runnable>(), factory, ThreadPoolExecutor.AbortPolicy()
        )
        try {
            // Without prestarting, execute() hands the first core tasks directly
            // to new workers instead of selecting them through the priority queue.
            executor.prestartAllCoreThreads()
        } catch (error: Throwable) {
            executor.shutdownNow()
            throw error
        }
    }

    fun <T> submit(priority: Int, computation: () -> T): CompletableFuture<T> {
        require(priority in 0..10) { "priority must be in [0, 10]" }
        val future = CompletableFuture<T>()
        synchronized(submissionLock) {
            check(!executor.isShutdown) { "pool is closed" }
            val job = PrioritizedJob(priority, sequence) {
                try { future.complete(computation()) }
                catch (error: Throwable) { future.completeExceptionally(error) }
                finally { Thread.interrupted() }
                Unit
            }
            sequence = sequence.add(BigInteger.ONE)
            // execute(), not submit(): submit() would wrap our comparable job
            // in a non-comparable FutureTask and break PriorityBlockingQueue.
            try { executor.execute(job) }
            catch (error: RejectedExecutionException) {
                throw IllegalStateException("pool is closed", error)
            }
        }
        return future
    }

    /** Nonblocking: reject new work and drain accepted tasks. Safe in a task. */
    fun close() { executor.shutdown() }

    /** Wait for executor termination. Only call from outside its workers. */
    fun shutdown() {
        check(!onWorker.get()) { "a worker cannot join its pool; use close()" }
        close()
        while (!executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)) {
            // No shutdown timeout: accepted computations must eventually return.
        }
    }

    private class PrioritizedJob(
        val priority: Int,
        val sequence: BigInteger,
        val computation: () -> Unit
    ) : Runnable, Comparable<PrioritizedJob> {
        override fun compareTo(other: PrioritizedJob): Int {
            val byPriority = other.priority.compareTo(priority)
            return if (byPriority != 0) byPriority else sequence.compareTo(other.sequence)
        }
        override fun run() = computation()
    }
}
