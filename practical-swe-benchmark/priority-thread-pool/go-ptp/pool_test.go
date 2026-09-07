package ptp

import (
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func await(t *testing.T, f *Future) (any, error) {
	t.Helper()
	select {
	case <-f.Done():
		return f.Await()
	case <-time.After(5 * time.Second):
		t.Fatal("future timed out")
		return nil, nil
	}
}

func TestPriorityFIFOErrorsAndDrain(t *testing.T) {
	p, _ := New(1)
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer p.Shutdown()
	defer once.Do(func() { close(release) })
	running, _ := p.Submit(0, func() (any, error) { close(started); <-release; return nil, nil })
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not start")
	}
	var order []int
	var futures []*Future
	for i, priority := range []int{0, 10, 5, 10, 0} {
		i := i
		f, err := p.Submit(priority, func() (any, error) { order = append(order, i); return i, nil })
		if err != nil {
			t.Fatal(err)
		}
		futures = append(futures, f)
	}
	failed, _ := p.Submit(7, func() (any, error) { return nil, errors.New("task failure") })
	panicked, _ := p.Submit(8, func() (any, error) { panic("boom") })
	p.Close()
	if _, err := p.Submit(0, func() (any, error) { return nil, nil }); err != ErrClosed {
		t.Fatal(err)
	}
	once.Do(func() { close(release) })
	await(t, running)
	for i, f := range futures {
		v, err := await(t, f)
		if err != nil || v != i {
			t.Fatalf("result: %v %v", v, err)
		}
	}
	if _, err := await(t, failed); err == nil {
		t.Fatal("missing error")
	}
	if _, err := await(t, panicked); err == nil {
		t.Fatal("missing panic")
	}
	p.Shutdown()
	p.Shutdown()
	if !reflect.DeepEqual(order, []int{1, 3, 2, 0, 4}) {
		t.Fatal(order)
	}
}

func TestParallelismAndValidation(t *testing.T) {
	if _, err := New(0); err == nil {
		t.Fatal("accepted zero workers")
	}
	p, _ := New(3)
	release := make(chan struct{})
	var once sync.Once
	defer p.Shutdown()
	defer once.Do(func() { close(release) })
	for _, priority := range []int{-1, 11} {
		if _, err := p.Submit(priority, func() (any, error) { return nil, nil }); err != ErrPriority {
			t.Fatal(err)
		}
	}
	if _, err := p.Submit(0, nil); err != ErrNilTask {
		t.Fatal(err)
	}
	started := make(chan struct{}, 8)
	var active, maximum atomic.Int32
	var futures []*Future
	for i := 0; i < 8; i++ {
		f, _ := p.Submit(0, func() (any, error) {
			n := active.Add(1)
			for old := maximum.Load(); n > old; old = maximum.Load() {
				if maximum.CompareAndSwap(old, n) {
					break
				}
			}
			started <- struct{}{}
			<-release
			active.Add(-1)
			return 42, nil
		})
		futures = append(futures, f)
	}
	for i := 0; i < 3; i++ {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("not parallel")
		}
	}
	if active.Load() != 3 {
		t.Fatal("wrong worker count")
	}
	once.Do(func() { close(release) })
	for _, f := range futures {
		await(t, f)
	}
	if maximum.Load() != 3 {
		t.Fatal("exceeded worker count")
	}
}
