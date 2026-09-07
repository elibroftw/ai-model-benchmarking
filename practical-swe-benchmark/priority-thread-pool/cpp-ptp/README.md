# C++ priority thread pool

Requires C++17, CMake 3.16+, and platform thread support. Standard library only;
the pool is a reusable header.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/ptp_example
# Or without CMake:
c++ -std=c++17 -pthread test.cpp -o /tmp/ptp-test && /tmp/ptp-test
```

```cpp
#include "priority_thread_pool.hpp"
PriorityThreadPool pool(4);
auto result = pool.submit(10, [] { return 12 * 12; });
int value = result.get(); // 144; task exceptions are rethrown here
```

`submit` accepts a nullary callable, including move-only captures, and returns
`std::future<T>` (including `void`). Capture arguments in the callable.
`close()` rejects new jobs and drains asynchronously. `shutdown()` joins workers;
the destructor also drains and joins (RAII). Multiple external shutdown calls
are serialized. Calling shutdown on a worker throws `std::logic_error`.

Keep the pool and all referenced captures alive until their tasks finish.
**Never destroy the pool inside one of its own tasks**; the destructor cannot
safely join itself. Exceptions in tasks are stored by `std::packaged_task`, so
workers survive. Dropping a future does not cancel its computation.

See the [shared scheduling contract](../README.md).
