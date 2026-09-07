# Go priority worker pool

Requires Go 1.20+. Standard library only.

```sh
go test -race -timeout 30s ./...
go run ./cmd/example
go vet ./...
```

```go
pool, err := ptp.New(4) // import ptp "example.com/priority-thread-pool"
if err != nil { panic(err) }
defer pool.Shutdown()
future, err := pool.Submit(10, func() (any, error) { return 12 * 12, nil })
if err != nil { panic(err) }
value, err := future.Await() // 144, nil
```

`Submit` returns a future or a validation/closed-pool error. `Await()` returns
the computation's value/error and is safe to call repeatedly or concurrently.
`Done()` exposes a completion channel for `select` with a caller's timeout or
context. Timing out a wait does not cancel work. Task panics become `*PanicError`,
including `panic(nil)` on older runtimes.

`Close()` rejects new tasks and requests a drain. `Shutdown()` closes and waits.
Go has no supported goroutine identity API, so **do not call Shutdown from a
pool task**; use Close instead. Do not call `runtime.Goexit` from a task: it
terminates the worker goroutine rather than returning/panicking normally. Pools
must be constructed with `New` and must not be copied.

These are a fixed number of worker **goroutines**, not pinned OS threads.
`GOMAXPROCS` controls CPU parallelism. The example pins only the caller's render
goroutine using `runtime.LockOSThread`, appropriate for graphics thread affinity.

See the [shared scheduling contract](../README.md).
