# Practical SWE Benchmark

In any complicated project, engineers will probably have to create their own reusable data structure and state machine that suits their needs.

The example that led to creating this project is a very simple one.

In any UI application, to avoid deadlocking, all UI updates are done single threaded. But what if we need to compute something beforehand. What if this wasn't just a GUI application, but did heavy rendering that ran on the GPU with multiple updates per second. Something like a game engine.

We can run computations on separate threads, and then once they are done, they can return something that would be used by the GPU (enqueued to the single render thread).

What if we wanted to optimized performance and have a heuristic on what is more important to be calculated? Introducing a priority thread pool. We have a n threads less than t tasks that need to be completed, with varying (0-10) priority values. What we want to do is to ensure that when we run work, the work with the highest priority will be taken first.

How I would implement this assuming a library/package does not exist.

- Heap + Thread pool

What we're looking for:

- Does the helper library already exist?
- Code maintainability
- Code succinctness

We will use pi agent.

First pass: We will use GPT-6 to do the work in all languages.

Once we got an idea how each implementation looks like in languages that I would consider working with, we can filter out the languages that seem like it leads to lower productivity for whatever the language offers.

Second pass: with a subset of languages or just one language, ask other AI models to implement it.

## Languages

- Rust
- C#
- Typescript
- Python
- C++
- Kotlin
- Go

### Results

Results are as follows:

Kotlin, C#, Python and C++ use the least lines of code.

- C# code criticism: not all syntactic sugar used
- Python code criticism: using deprecated imports. Language criticism: lol.
- Kotlin language criticism: "workerNumber.getAndIncrement", "preStartAllCoreThreads", "synchronized", make it harder to be a competent developer
- C++ language criticism: I have no desire to pick C++, I was just curious about the implementation versus rust

Typescript: Needed to tell the model to use piscina after it implemented "from scratch", and even then it needed to implement way too much from scratch like the queue.

Rust: model didn't mention tokio, so who knows.

So I'm still bullish on C# as a backend language. For systems, I guess Rust wins on new projects just because of memory security. I will say though that using Rust doesn't mean secure, so it's important to still keep that in mind. The tooling is much better in Rust as well, so there's no headaches.
