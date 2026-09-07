#include "priority_thread_pool.hpp"
#include <chrono>
#include <iostream>
#include <set>

using namespace std::chrono_literals;
void check(bool ok) { if (!ok) throw std::runtime_error("test failed"); }
template<class E, class F> void throws(F fn) {
    try { fn(); } catch (const E&) { return; }
    throw std::runtime_error("expected exception");
}

int main() {
    throws<std::invalid_argument>([] { PriorityThreadPool pool(0); });
    PriorityThreadPool pool(1);
    throws<std::invalid_argument>([&] { pool.submit(-1, [] {}); });
    throws<std::invalid_argument>([&] { pool.submit(11, [] {}); });
    std::promise<void> started, release;
    auto gate = release.get_future().share();
    auto running = pool.submit(0, [&] { started.set_value(); check(gate.wait_for(5s) == std::future_status::ready); });
    check(started.get_future().wait_for(5s) == std::future_status::ready);
    std::vector<int> order;
    std::vector<std::future<int>> results;
    int priorities[] = {0, 10, 5, 10, 0};
    for (int i = 0; i < 5; ++i)
        results.push_back(pool.submit(priorities[i], [&, i] { order.push_back(i); return i; }));
    auto failed = pool.submit(7, []() -> int { throw std::runtime_error("task failure"); });
    pool.close();
    throws<std::runtime_error>([&] { pool.submit(0, [] {}); });
    release.set_value();
    pool.shutdown();
    pool.shutdown();
    running.get();
    check(order == std::vector<int>({1, 3, 2, 0, 4}));
    for (int i = 0; i < 5; ++i) check(results[i].get() == i);
    throws<std::runtime_error>([&] { failed.get(); });

    PriorityThreadPool parallel(3);
    std::promise<void> release_all;
    auto all_gate = release_all.get_future().share();
    std::array<std::promise<void>, 3> entered;
    std::vector<std::future<std::thread::id>> ids;
    for (int i = 0; i < 3; ++i)
        ids.push_back(parallel.submit(0, [&, i] {
            entered[i].set_value();
            check(all_gate.wait_for(5s) == std::future_status::ready);
            return std::this_thread::get_id();
        }));
    for (auto& signal : entered) check(signal.get_future().wait_for(5s) == std::future_status::ready);
    release_all.set_value();
    std::set<std::thread::id> unique;
    for (auto& id : ids) unique.insert(id.get());
    check(unique.size() == 3);
    auto self_join = parallel.submit(0, [&] { parallel.shutdown(); });
    throws<std::logic_error>([&] { self_join.get(); });
    auto move_only = parallel.submit(0, [p = std::make_unique<int>(42)] { return *p; });
    check(move_only.get() == 42);
    std::cout << "All tests passed\n";
}
