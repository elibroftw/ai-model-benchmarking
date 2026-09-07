#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class PriorityThreadPool {
public:
    explicit PriorityThreadPool(int workers) {
        if (workers <= 0) throw std::invalid_argument("workers must be positive");
        threads_.reserve(static_cast<std::size_t>(workers));
        try {
            for (int i = 0; i < workers; ++i)
                threads_.emplace_back([this] { run(); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    // The owner must outlive tasks; do not destroy the pool on one of its workers.
    ~PriorityThreadPool() { shutdown(); }
    PriorityThreadPool(const PriorityThreadPool&) = delete;
    PriorityThreadPool& operator=(const PriorityThreadPool&) = delete;

    template<class F>
    auto submit(int priority, F&& fn) -> std::future<std::invoke_result_t<F>> {
        if (priority < 0 || priority > 10)
            throw std::invalid_argument("priority must be in [0, 10]");
        using T = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<T()>>(std::forward<F>(fn));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) throw std::runtime_error("pool is closed");
            queues_[priority].emplace_back([task] { (*task)(); });
            ++pending_;
        }
        ready_.notify_one();
        return result;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

    void shutdown() {
        if (current_pool_ == this)
            throw std::logic_error("a worker cannot join its pool; use close()");
        close();
        std::lock_guard<std::mutex> lock(join_mutex_);
        for (auto& thread : threads_)
            if (thread.joinable()) thread.join();
    }

private:
    void run() {
        current_pool_ = this;
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return closed_ || pending_ != 0; });
                if (pending_ == 0) break;
                for (int p = 10; p >= 0; --p) {
                    if (!queues_[p].empty()) {
                        job = std::move(queues_[p].front());
                        queues_[p].pop_front();
                        --pending_;
                        break;
                    }
                }
            }
            job(); // packaged_task stores user exceptions in its future
        }
        current_pool_ = nullptr;
    }

    inline static thread_local PriorityThreadPool* current_pool_ = nullptr;
    std::mutex mutex_, join_mutex_;
    std::condition_variable ready_;
    std::array<std::deque<std::function<void()>>, 11> queues_;
    std::size_t pending_ = 0;
    bool closed_ = false;
    std::vector<std::thread> threads_;
};
