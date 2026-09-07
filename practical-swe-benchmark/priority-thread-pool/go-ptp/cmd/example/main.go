package main

import (
	ptp "example.com/priority-thread-pool"
	"fmt"
	"runtime"
)

func main() {
	// Graphics APIs may require OS-thread affinity, not just one goroutine.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool, err := ptp.New(2)
	if err != nil {
		panic(err)
	}
	defer pool.Shutdown()
	var meshes []*ptp.Future
	for _, priority := range []int{1, 10, 5} {
		size := priority * 100
		mesh, err := pool.Submit(priority, func() (any, error) {
			vertices := make([]int, size)
			for i := range vertices {
				vertices[i] = i * i
			}
			return vertices, nil
		})
		if err != nil {
			panic(err)
		}
		meshes = append(meshes, mesh)
	}
	for _, mesh := range meshes {
		vertices, err := mesh.Await()
		if err != nil {
			panic(err)
		}
		fmt.Println("Upload mesh to GPU:", len(vertices.([]int)), "vertices")
	}
}
