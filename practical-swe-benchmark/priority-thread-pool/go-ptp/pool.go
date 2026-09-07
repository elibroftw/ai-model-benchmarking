// Package ptp runs priority jobs on a fixed number of worker goroutines.
package ptp

import (
	"errors"
	"fmt"
	"sync"
)

var (
	ErrClosed   = errors.New("pool is closed")
	ErrPriority = errors.New("priority must be in [0, 10]")
	ErrNilTask  = errors.New("task must not be nil")
)

// PanicError reports a task panic without losing its worker.
type PanicError struct{ Value any }

func (e *PanicError) Error() string { return fmt.Sprintf("task panicked: %v", e.Value) }

// Future can be awaited repeatedly by any number of goroutines.
type Future struct {
	done  chan struct{}
	value any
	err   error
}

func (f *Future) Done() <-chan struct{} { return f.done }
func (f *Future) Await() (any, error) {
	<-f.done
	return f.value, f.err
}

type job struct {
	fn     func() (any, error)
	future *Future
	next   *job
}

type queue struct{ head, tail *job }

func (q *queue) push(j *job) {
	if q.tail == nil {
		q.head = j
	} else {
		q.tail.next = j
	}
	q.tail = j
}
func (q *queue) pop() *job {
	j := q.head
	if j != nil {
		q.head = j.next
		j.next = nil
		if q.head == nil {
			q.tail = nil
		}
	}
	return j
}

// Pool must not be copied. Queues and lifecycle share one lock so Submit and
// Close are linearizable. User functions never execute while holding this lock.
type Pool struct {
	mu      sync.Mutex
	ready   *sync.Cond
	queues  [11]queue
	pending int
	closed  bool
	workers sync.WaitGroup
}

func New(workers int) (*Pool, error) {
	if workers <= 0 {
		return nil, errors.New("workers must be positive")
	}
	p := &Pool{}
	p.ready = sync.NewCond(&p.mu)
	p.workers.Add(workers)
	for i := 0; i < workers; i++ {
		go p.run()
	}
	return p, nil
}

func (p *Pool) Submit(priority int, fn func() (any, error)) (*Future, error) {
	if priority < 0 || priority > 10 {
		return nil, ErrPriority
	}
	if fn == nil {
		return nil, ErrNilTask
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, ErrClosed
	}
	f := &Future{done: make(chan struct{})}
	p.queues[priority].push(&job{fn: fn, future: f})
	p.pending++
	p.ready.Signal()
	return f, nil
}

// Close rejects new tasks and wakes workers to drain the queue; it never waits.
func (p *Pool) Close() {
	p.mu.Lock()
	p.closed = true
	p.ready.Broadcast()
	p.mu.Unlock()
}

// Shutdown closes and waits. Never call it from a task in this pool: use Close.
func (p *Pool) Shutdown() {
	p.Close()
	p.workers.Wait()
}

func (p *Pool) run() {
	defer p.workers.Done()
	for {
		p.mu.Lock()
		for p.pending == 0 && !p.closed {
			p.ready.Wait()
		}
		if p.pending == 0 {
			p.mu.Unlock()
			return
		}
		var j *job
		for priority := 10; priority >= 0; priority-- {
			if j = p.queues[priority].pop(); j != nil {
				break
			}
		}
		p.pending--
		p.mu.Unlock()
		execute(j)
	}
}

func execute(j *job) {
	completed := false
	defer func() {
		if !completed {
			j.future.err = &PanicError{Value: recover()}
		}
		close(j.future.done)
	}()
	j.future.value, j.future.err = j.fn()
	completed = true
}
