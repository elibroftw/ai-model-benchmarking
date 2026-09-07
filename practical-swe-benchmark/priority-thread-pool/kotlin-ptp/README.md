# Kotlin priority thread pool

Requires **JDK 17–24** (tested with JDK 21 on Fedora). The checked-in Gradle
8.14.3 wrapper downloads Gradle and the Kotlin 2.2.21 compiler automatically;
you do not need to install Gradle or `kotlinc`. The first build requires internet
access. Runtime code uses only the Kotlin and Java standard libraries; tests
use Kotlin Test and JUnit 5.

## Run tests and main

From this project directory:

```sh
./gradlew test       # Run the JUnit test suite
./gradlew run        # Run ExampleKt, the compute/render demonstration
./gradlew build      # Compile, test, and package the application
```

From the repository root, first run:

```sh
cd practical-swe-benchmark/priority-thread-pool/kotlin-ptp
```

Gradle skips tests when their inputs are unchanged. To force another run, use
`./gradlew test --rerun-tasks`. The HTML test report is at
`build/reports/tests/test/index.html`.

Sources follow the standard Gradle layout:

- `src/main/kotlin/PriorityThreadPool.kt`: reusable pool
- `src/main/kotlin/Example.kt`: application entry point
- `src/test/kotlin/PriorityThreadPoolTest.kt`: JUnit tests

## Implementation

Uses Java's `ThreadPoolExecutor` with equal core/maximum sizes and a
`PriorityBlockingQueue`. Comparable job wrappers sort by descending priority,
then by a monotonically increasing `BigInteger` sequence for stable FIFO ties
without counter overflow. Sequence allocation and enqueue are serialized so
ties reflect actual enqueue order even with concurrent producers.

All core workers are prestarted to avoid the executor's initial direct-dispatch
path. Jobs go through `execute()`, not `submit()`, because the latter would wrap
them in non-comparable `FutureTask`s. The standard executor owns worker lifecycle,
queue waiting, shutdown, and termination; the wrapper retains the existing
`CompletableFuture` API.

```kotlin
val pool = PriorityThreadPool(4)
try {
    val result = pool.submit(10) { 12 * 12 }
    println(result.get()) // 144
} finally {
    pool.shutdown()
}
```

`submit<T>(priority, computation)` returns `CompletableFuture<T>`; `Unit` works
for void jobs. Exceptions complete the future exceptionally and do not remove
a worker (`get()` wraps them in `ExecutionException`). These are synchronous
computations on dedicated JVM threads, not suspend functions.

`close()` invokes the executor's nonblocking `shutdown()`, rejecting new tasks
while allowing queued jobs to finish. The pool's own `shutdown()` additionally
waits with `awaitTermination()` and rejects calls from a worker. Interruption
of that wait propagates to the caller; already accepted work continues draining.
The pool is not `AutoCloseable`; use `try/finally` as above.
Cancelling a returned `CompletableFuture` changes that future's state but does
not remove or interrupt the computation. Callback methods such as `thenAccept`
may run on a worker; use your UI framework's dispatcher for render updates.

See the [shared scheduling contract](../README.md).
