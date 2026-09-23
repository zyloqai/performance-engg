# AI Systems & Performance Engineering — 30-Day Syllabus

**Project:** AI Systems Performance Lab  
**Author / learner:** Sankalp  
**Recorded sprint window:** 21 September–20 October 2026  
**Approach:** Questions first. Experiments next. Evidence before claims.  
**Status:** Learning roadmap, not a record of 30 completed days.

> **The central question:** When an AI request is slow, expensive, incorrect, or unreliable, can I identify the responsible layer, design a useful experiment, and defend a better decision?

This repository develops a connected understanding of **Linux, C++, parallel programming, GPU architecture, CUDA, transformer inference, serving systems, reliability, and hardware–software trade-offs**. The purpose is genuine technical depth: being able to build, investigate, explain, and eventually teach these systems—not merely use their APIs or prepare interview answers.

Thirty days is the first focused cycle, not a claim of complete mastery or a guarantee of a Staff/Principal role. A smaller body of correct, reproducible work is preferable to thirty unchecked folders.

**Progress note:** Day-1 tensor, tokenization, prefill/decode, and exploratory context-length experiments have been reported and documented. The benchmark needs revision, and some environment metadata remains uncaptured. Days 2–30 are planned until their evidence is linked. Read the [Day-1 lab record][DAY01-NOTES] and [Day-1 curiosity questions][DAY01-QUESTIONS] for the existing results and limitations.

## Contents

- [1. Learning contract](#learning-contract)
- [2. The connected skill model](#skill-model)
- [3. Flagship project and scope](#flagship-project)
- [4. Thirty days at a glance](#calendar)
- [5. Daily syllabus](#daily-syllabus)
- [6. Benchmark and evidence standards](#evidence-standard)
- [7. Repository and publication workflow](#publication)
- [8. Resources and hardware budget](#resources)
- [9. Final assessment and the next year](#next-cycle)

---

<a id="learning-contract"></a>
## 1. Learning contract

### The learning loop

```text
Ask why → predict → study the minimum needed → implement → test correctness
        → measure → inspect evidence → challenge the explanation
        → change one thing → re-measure → document → obtain review
```

For each experiment, distinguish **observation**, **calculation**, **hypothesis**, and **untested design**. “The latency decreased” is an observation. “Memory traffic caused the decrease” requires additional evidence.

### What counts as understanding?

A topic is becoming a working skill when I can explain its purpose, implement a small version, test it, measure it, diagnose a failure, and defend its trade-offs without relying on a generated explanation. The goal is not to memorize every tool flag.

### Prerequisites and pacing

The starting point is practical Python, basic terminal use, and the completed Day-1 GPU experiment. C++ and CUDA depth must be assessed rather than assumed. Begin with an honest gap list; revisit it every review day.

The existing Excel tracker budgets **10 core hours on build days, up to 2 optional hours, and 4 hours on Days 7, 14, 21, and 28**. These are planning ceilings, not a requirement to work through exhaustion. Protect sleep, meals, movement, and attention. Optional work is not debt. If a foundation is weak, reduce stretch scope before adding hours.

The day numbers describe a learning sequence. If work slips, record the actual completion date rather than backdating results or declaring unfinished work complete. This file keeps the tracker’s day order; it does not replace its actual-hours and status fields.

### Focus rules

- **One specialization:** AI systems and performance engineering.
- **One flagship repository:** related experiments belong here.
- **One primary model:** retain the Day-1 model unless an explicit compatibility or capacity issue requires a documented change.
- **One primary serving engine:** introduce vLLM after the underlying execution path is understood.
- **Security and reproducibility start on Day 1.** Later dedicated days deepen them; they do not introduce them for the first time.
- **A tool is admitted to the project only when it answers a current question.** Kubernetes, another GPU vendor, a second serving engine, and new business applications are not automatic additions.

---

<a id="skill-model"></a>
## 2. The connected skill model

We learn downward to understand execution, then upward to understand user experience.

| Layer | The question it helps answer |
|---|---|
| User and service objectives | What response time, quality, reliability, and cost are acceptable? |
| Request handling and scheduling | Who waits, who runs, and what happens during overload? |
| Model execution | What work happens during prefill, cached decode, and token selection? |
| Framework and runtime | Which operations and kernels are dispatched, and by whom? |
| Kernels and libraries | How are mathematical operations mapped to parallel work? |
| GPU architecture and memory | What do warps, compute units, registers, shared memory, caches, and VRAM limit? |
| CPU and operating system | Where do allocation, dispatch, syscalls, threads, and data preparation consume time? |
| Interconnects and distributed execution | What must move between devices or machines, and what does that movement cost? |
| Deployment and operations | Can the system restart, reject unsafe work, recover, and produce useful telemetry? |
| Hardware–software co-design | Which software or hardware change would address the measured limit? |

Keep **access**, **container setup**, and **computation** separate:

```text
Access:          laptop → SSH → remote shell
Container setup: Docker + NVIDIA integration → process with GPU access
Computation:     Python/framework → CUDA operations → host driver → GPU
```

NVIDIA Container Toolkit enables the container environment; it is not a separate arithmetic engine traversed for every multiplication. The containerized process still uses the host’s kernel and GPU driver. [Architecture reference][NVIDIA-CONTAINERS].

### Three persistent questions

**Where is the data?** Disk, CPU memory, GPU memory, cache, or another machine?

**Who is doing the work?** A CPU thread, GPU kernel, network device, or storage system?

**What is waiting for what?** Dependencies and resource contention often matter more than the name of the framework.

---

<a id="flagship-project"></a>
## 3. Flagship project and scope

### AI Systems Performance Observatory

Build an **enterprise-oriented reference implementation** that runs controlled inference workloads, preserves raw measurements, investigates bottlenecks, and documents operating limits. It is not called production-ready merely because it has a container or a dashboard.

```text
Controlled workload client
        │ requests and client timestamps
        ▼
Private serving endpoint
        │ validation, request identity, admission and bounded waiting
        ▼
Serving engine and scheduler
        │ prefill / decode work and request-specific state
        ▼
Shared model weights + managed KV cache
        │ framework / kernels / CUDA / host driver
        ▼
GPU execution

Across the path: metrics, traces, correctness checks, failure tests,
versioned configuration, raw result files, and a reproducible report.
```

The drawing describes the target system. Components become “implemented” only after their code and tests exist. We use the serving engine’s capabilities where appropriate rather than rebuilding an entire inference runtime.

### Starting environment—not a recommendation to install the latest of everything

Day 1 used an **RTX A6000**, the `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` image, PyTorch `2.7.1+cu128` during the tensor test, and `Qwen/Qwen2.5-7B-Instruct` with BF16 requested. These are recorded historical observations; capture the final package inventory and model revision before formal comparisons. The RTX A6000 is not the RTX 6000 Ada.

Maintain three environment definitions as needed:

| Environment | Purpose | Rule |
|---|---|---|
| Inference-learning image | Preserve the Day-1 Transformers experiments | Capture versions and image digest; do not overwrite the baseline silently. |
| CUDA development image | Compile C++/CUDA kernels and use development tools | Add a compatible Toolkit/compiler here; a runtime-only image need not contain `nvcc`. |
| Serving image | Run a mutually compatible vLLM/PyTorch/CUDA stack | Validate separately on Day 15; do not force a new engine into the old container through untracked package upgrades. |

A configuration change is recorded as an experimental variable. Capture GPU identity, driver, software versions, model/tokenizer commits, attention backend, precision, workload, and commands. The [Day-1 record][DAY01-NOTES] lists what is still missing.

### Release gates

| Checkpoint | What must exist before declaring the milestone complete |
|---|---|
| **v0.1 — Day 7** | Documented setup, corrected basic benchmark, Linux/C++ experiments, tested queue, and first CUDA kernel. |
| **v0.2 — Day 14** | Correct kernel labs, reproducible timings, profiler evidence where available, and a successful or unsuccessful optimization explained honestly. |
| **v0.3 — Day 21** | A private serving baseline and controlled comparisons for scheduling, memory, caching, and one supported optimization method. |
| **v1.0 — Day 30** | Rebuild instructions, tests, failure behavior, capacity assumptions, final report, and explicit limitations. |

Multi-GPU execution is gated on access and budget. Kubernetes, a new compiler backend, and a complete AMD port are stretch work—not requirements for the first release.

---

<a id="calendar"></a>
## 4. Thirty days at a glance

This is the same core sequence as `AI_Systems_30_Day_Tracker_21_Sep_2026.xlsx`. Dates are the original plan, not assertions of completion.

| Day | Planned date | Topic | Evidence target |
|---|---|---|---|
| [01](#day-01) | 21 Sep | Baseline and execution map | Layered execution map and exploratory lab record |
| [02](#day-02) | 22 Sep | Linux and program execution | Annotated program-execution investigation |
| [03](#day-03) | 23 Sep | CPU memory hierarchy | Size/stride experiment and raw timings |
| [04](#day-04) | 24 Sep | C++ ownership and lifetime | Tested resource-owning benchmark utility |
| [05](#day-05) | 25 Sep | Concurrency and backpressure | Bounded queue with stress and shutdown tests |
| [06](#day-06) | 26 Sep | CUDA execution model | Vector-add correctness and timing breakdown |
| [07](#day-07) | 27 Sep | Review 1 and v0.1 | Reproduction, engineering note, prioritized gaps |
| [08](#day-08) | 28 Sep | GPU memory access | Naive/tiled transpose comparison |
| [09](#day-09) | 29 Sep | Reduction and synchronization | Correct reduction variants and error analysis |
| [10](#day-10) | 30 Sep | GEMM, reuse and tiling | Tiled matrix multiplication versus a library |
| [11](#day-11) | 01 Oct | RMSNorm and numerical correctness | Normalization kernel with justified tolerances |
| [12](#day-12) | 02 Oct | Profile before optimizing | Annotated traces and competing hypotheses |
| [13](#day-13) | 03 Oct | Roofline and performance limits | Predicted limits versus measured performance |
| [14](#day-14) | 04 Oct | Review 2 and v0.2 | Kernel portfolio and outside critique request |
| [15](#day-15) | 05 Oct | Inference anatomy and serving baseline | Private endpoint and client-observed metrics |
| [16](#day-16) | 06 Oct | KV cache and capacity | Memory estimate and measured capacity curve |
| [17](#day-17) | 07 Oct | Scheduling and batching | Multi-user latency/throughput trade-off |
| [18](#day-18) | 08 Oct | Prefix caching | Explicit hit/miss experiment |
| [19](#day-19) | 09 Oct | Quantization and quality | Supported precision comparison with evaluation |
| [20](#day-20) | 10 Oct | CUDA Graphs and launch overhead | Capture/replay experiment with matched controls |
| [21](#day-21) | 11 Oct | Review 3 and v0.3 | Defensible inference case study |
| [22](#day-22) | 12 Oct | Reliability, overload and security | Failure matrix and recovery evidence |
| [23](#day-23) | 13 Oct | Reproducible deployment | Clean rebuild and state-recovery test |
| [24](#day-24) | 14 Oct | Distributed communication — gated | Actual collective test or labeled design study |
| [25](#day-25) | 15 Oct | Networking and scale-out reasoning | Topology and communication-cost model |
| [26](#day-26) | 16 Oct | Capacity and economics | SLO-based capacity/cost report |
| [27](#day-27) | 17 Oct | Hardware–software co-design | Two-page evidence-based architecture memo |
| [28](#day-28) | 18 Oct | Review 4 and upstream contribution | Review response and useful upstream artifact |
| [29](#day-29) | 19 Oct | Mock loop and public explanation | Independent assessment and talk/article draft |
| [30](#day-30) | 20 Oct | Final release and next cycle | v1.0, retrospective, one next focus |

---

<a id="daily-syllabus"></a>
## 5. Daily syllabus

### Phase 1 — Understand the machine before optimizing it

**Days 1–7:** Connect a high-level AI request to processes, memory, ownership, concurrency, and the first manually written GPU program.

<a id="day-01"></a>
### Day 01 — Baseline and execution map

> **Question:** What actually happens between submitting text and computing the next token on a GPU?

**Why it matters:** A successful `nvidia-smi` query, a running container, and a correct inference result answer different questions. Separating those checks prevents debugging the wrong layer.

**Study:** Driver versus CUDA Toolkit/runtime; container setup versus computation; CPU tensors versus VRAM; token IDs and embeddings; model weights; prefill; cached decode.

**Build and inspect:** Preserve the existing tensor and transformer scripts. Document SSH, package-lock, and container-access issues as diagnostic cases. Archive the original context sweep before fixing its warm-up, allocator, and object-lifetime problems. Capture missing environment metadata.

**Evidence:** The [lab notes][DAY01-NOTES], [curiosity workbook][DAY01-QUESTIONS], actual source files, raw observations, and an execution map. Existing numbers remain exploratory, not serving latency or proof of a bottleneck.

**Expert challenge:** How could `nvidia-smi` work while a CUDA application fails? Which test would distinguish those layers?

**Reading:** [PyTorch CUDA semantics][PYTORCH-CUDA], [container architecture][NVIDIA-CONTAINERS], [KV-cache explanation][HF-CACHE].

<a id="day-02"></a>
### Day 02 — Linux and program execution

> **Question:** After I type a command, what makes instructions actually run?

**Why it matters:** The GPU is only one resource. Model loading, file access, tokenization, and kernel dispatch have CPU/OS behavior that can limit the complete system.

**Study:** Programs versus processes; process/thread IDs; compilation and linking; virtual address spaces; syscalls; scheduling; permissions; exit codes. Connect the Day-1 APT lock to coordination over shared state.

**Build and inspect:** Compile one small C++ program that allocates memory and performs a deterministic calculation. Observe its process and `/proc` entries; inspect syscalls and timing where tools and permissions allow. Explain startup separately from useful work.

**Evidence:** Build command, compiler version, checked result, annotated observations, and a request-to-process diagram. Record unavailable tools rather than infer their output.

**Expert challenge:** If two processes use the same virtual address, are they necessarily accessing the same physical memory?

**Reading:** [CS:APP][CSAPP], [Linux `/proc` documentation][LINUX-PROC].

<a id="day-03"></a>
### Day 03 — CPU memory hierarchy

> **Question:** Why can two loops doing comparable arithmetic take very different amounts of time?

**Why it matters:** Data movement is a first-class part of computation. This experiment builds the intuition later needed for GPU coalescing, tiling, and model-weight traffic.

**Study:** Cache lines, locality, working sets, bandwidth versus latency, prefetching, and compiler transformations. Treat virtual-machine topology as something to inspect, not assume.

**Build and measure:** Sweep working-set size and access stride. Keep the number of accesses explicit, preserve a checked checksum so the compiler cannot discard the work, and hold compiler options constant. Report which bytes are logical accesses versus estimated traffic.

**Evidence:** Raw repetitions, size/stride plot, checksum, build settings, and competing explanations for changes in runtime.

**Expert challenge:** Did changing stride also change the amount of useful work? How would that invalidate the comparison?

**Reading:** [CS:APP memory-hierarchy material][CSAPP].

<a id="day-04"></a>
### Day 04 — C++ ownership and lifetime

> **Question:** Who owns each allocation, and what happens when execution exits unexpectedly?

**Why it matters:** Incorrect lifetime management can cause crashes or retained memory; repeated unnecessary allocations can also affect performance. The Day-1 retained Python cache references provide a concrete motivation, although C++ uses different lifetime mechanisms.

**Study:** RAII—tying resource cleanup to object lifetime—stack/heap distinctions, references, smart pointers, moves, exceptions, and error paths.

**Build and test:** Create a small benchmark utility that owns buffers and writes structured results. Exercise normal completion, early return, failure, and move behavior. Use compiler warnings and supported sanitizers; simplify interfaces before pursuing clever abstractions.

**Evidence:** Tests, ownership diagram, sanitizer findings and fixes, and documented buffer-reuse behavior.

**Expert challenge:** Can a program have no permanent leak yet still hold enough unnecessary memory to reduce serving capacity?

**Reading:** [C++ Core Guidelines: resource management and errors][CPP-GUIDELINES].

<a id="day-05"></a>
### Day 05 — Concurrency and backpressure

> **Question:** What should happen when requests arrive faster than workers finish them?

**Why it matters:** Multi-user inference begins with scheduling and resource limits, not with loading one model per user. A bounded queue makes overload policy explicit.

**Study:** Concurrency versus parallelism; mutexes; condition variables; races; contention; bounded queues; shutdown; backpressure. An atomic variable alone does not make an entire algorithm correct.

**Build and measure:** Implement a C++ producer–consumer queue. Simulate two users with different task durations on the CPU; log arrival, service start, and completion. Test full/empty queues, multiple producers, and shutdown without lost or duplicated tasks.

**Evidence:** Tests, queue timeline, wait-versus-service measurements, and the chosen full-queue policy. This is a scheduling simulation, not a GPU-serving result.

**Expert challenge:** Could a larger queue increase completed work while making the service unacceptable to users?

**Reading:** [C++ Core Guidelines: concurrency][CPP-GUIDELINES].

<a id="day-06"></a>
### Day 06 — CUDA execution model

> **Question:** Why can a GPU lose to a CPU on a small calculation?

**Why it matters:** Launching work and moving data have costs. GPU performance must be evaluated at the boundary the application actually cares about.

**Study:** Host/device code; grids, blocks, threads, and warps; streams; asynchronous execution; explicit synchronization; allocation and transfers. Map logical threads onto the programming model before memorizing hardware counts.

**Build and measure:** Add a compatible CUDA development environment with `nvcc`, keeping the Day-1 runtime image unchanged. Implement bounds-checked vector addition against a CPU reference. Sweep sizes and distinguish allocation, transfer, kernel, and end-to-end time. Check API and kernel-launch errors.

**Evidence:** Correctness tests, toolchain manifest, device properties, and a timing breakdown with warm-up and units.

**Expert challenge:** Is a faster kernel useful if transferring its input and output dominates the application?

**Reading:** [CUDA Programming Guide][CUDA-GUIDE], [CUDA Best Practices][CUDA-BP].

<a id="day-07"></a>
### Day 07 — Review 1 and v0.1

> **Question:** Could someone reproduce the first week without reading our chat history?

**Why it matters:** Knowledge becomes portable when the environment, source, assumptions, and instructions survive outside a running terminal.

**Study:** No new domain today. Retrieve the execution map from memory and revisit the weakest explanation.

**Build and verify:** Re-run the setup and selected tests from the README. Finish the corrected Day-1 baseline: warm both prefill and cached decode, manage references explicitly, and preserve raw timings. Compare cached and full-context next-token calculations under documented tolerances before trusting measurements.

**Evidence:** `v0.1` only after its gate passes; Week-1 engineering note; ten-minute explanation; missing-metadata list; one prioritized weakness. Request feedback without assuming a reviewer will respond immediately.

**Expert challenge:** Which statement in the report would you remove because the evidence does not support it?

**Reading:** Own code and results; [PyTorch timing/memory semantics][PYTORCH-CUDA].

---

### Phase 2 — Explain and improve GPU work

**Days 8–14:** Learn memory access, synchronization, reuse, numerical correctness, profiling, and performance bounds through a small kernel portfolio.

<a id="day-08"></a>
### Day 08 — GPU memory access

> **Question:** How can changing addresses make the same mathematical operation faster?

**Why it matters:** Arithmetic throughput is not useful when data movement prevents the compute units from being fed efficiently. An access-pattern experiment makes that constraint tangible.

**Study:** Global-memory transactions; coalescing; shared-memory tiles; bank conflicts; alignment; edge handling. Distinguish shared memory from GPU-wide VRAM.

**Build and measure:** Implement naive and tiled transpose. Include rectangular and non-tile-aligned matrices. Keep dtype, dimensions, input residency, and timing protocol matched. Define effective bandwidth from explicitly counted useful bytes, not as proof of actual DRAM traffic.

**Evidence:** Correctness tests, address-pattern sketches, raw timings, and the conditions where tiling helps or does not.

**Expert challenge:** If a tile reduces global-memory traffic but adds synchronization, when could it lose?

**Reading:** [CUDA memory and synchronization model][CUDA-GUIDE], [memory optimization guidance][CUDA-BP].

<a id="day-09"></a>
### Day 09 — Reduction and synchronization

> **Question:** How can thousands of threads combine results without racing or hiding numerical errors?

**Why it matters:** Reductions appear in normalization, attention, and many numerical algorithms. They force us to reason about both coordination and floating-point behavior.

**Study:** Partial results, barriers, warp-level operations, multi-stage reduction, and non-associativity of floating-point addition.

**Build and test:** Implement a reference and two GPU reduction variants. Include empty inputs with defined behavior, odd sizes, short inputs, and large inputs. Compare error and timing; use appropriate sanitizer checks where supported. Do not treat passing random tests as a proof of race freedom.

**Evidence:** Edge-case suite, tolerance rationale, synchronization explanation, and measured comparisons.

**Expert challenge:** Can a faster parallel sum legitimately differ from a serial sum—and when would the difference be unacceptable?

**Reading:** [CUDA Programming Guide][CUDA-GUIDE], [Compute Sanitizer][COMPUTE-SANITIZER].

<a id="day-10"></a>
### Day 10 — GEMM, reuse and tiling

> **Question:** Why does matrix multiplication benefit so much from reusing the same data?

**Why it matters:** GEMM—general matrix multiplication—is a useful bridge between model mathematics and hardware execution. It exposes the difference between doing arithmetic and feeding arithmetic efficiently.

**Study:** Matrix shapes, tiling, arithmetic intensity, registers/shared memory, resource pressure, and a first look at Tensor Core paths.

**Build and measure:** Compare a naive kernel, a tiled kernel, and an optimized library for matched shapes and precision. Record layout, accumulation behavior, and TF32 or other precision settings where applicable. Separate device-resident execution from transfers.

**Evidence:** Verified outputs, timing table, reuse sketch, and an honest explanation of the remaining library-performance gap.

**Expert challenge:** Are you comparing implementations of the same numerical contract, or giving the library a different precision budget?

**Reading:** [CUDA programming model][CUDA-GUIDE], [PyTorch precision controls][PYTORCH-CUDA].

<a id="day-11"></a>
### Day 11 — RMSNorm and numerical correctness

> **Question:** Why can a short normalization formula still deserve a specialized kernel?

**Why it matters:** Not every important operation is a large matrix multiplication. A reduction followed by elementwise work can reveal avoidable intermediate traffic and launch overhead.

**Study:** Root-mean-square normalization; epsilon; accumulation precision; row width; intermediate tensors; fusion opportunities and their boundaries.

**Build and test:** Write a small CUDA RMSNorm and an independent reference. Test representative widths, non-aligned sizes, and challenging values. Establish justified tolerances before measuring. Attempt one fused variant only after the simple version is correct.

**Evidence:** Tests, error summaries, raw timings, and a hypothesis about traffic versus computation. Do not claim the kernel speeds up the model unless it is actually integrated and evaluated there.

**Expert challenge:** Did reducing kernel count remove useful work or accidentally change the mathematical result?

**Reading:** [CUDA programming and numerical behavior][CUDA-GUIDE], [Compute Sanitizer][COMPUTE-SANITIZER].

<a id="day-12"></a>
### Day 12 — Profile before optimizing

> **Question:** Is the GPU slow, or is it waiting for something else?

**Why it matters:** Latency alone does not identify its cause. We now connect measured time to CPU dispatch, kernels, copies, synchronization, and dependencies.

**Study:** System timelines versus kernel counters; NVTX ranges; profiler perturbation; replay; permissions; launch gaps; overlap. A utilization percentage is not a complete diagnosis.

**Build and inspect:** Capture a Nsight Systems trace of one inference step and one kernel lab. Use Nsight Compute on a narrowly selected kernel where counters are available. Annotate two observations and two possible explanations before changing code. Collect normal benchmark timing separately from profiling.

**Evidence:** Capture commands, trace artifacts, annotated screenshots, permissions/limits, and one falsifiable bottleneck hypothesis.

**Expert challenge:** Which observation would show that your favorite optimization targets the wrong layer?

**Reading:** [Nsight Systems][NSYS], [Nsight Compute][NCU].

<a id="day-13"></a>
### Day 13 — Roofline and performance limits

> **Question:** How fast could this workload plausibly run, and what would cap the improvement?

**Why it matters:** A bound helps prioritize engineering work. It prevents spending days accelerating a small component that cannot materially improve the complete request.

**Study:** Arithmetic intensity—operations per byte—compute and bandwidth ceilings, Amdahl’s law, theoretical versus sustained bandwidth, and assumptions about traffic.

**Build and compare:** Derive a simple roofline estimate for one kernel and an Amdahl bound for one pipeline. Show units and the counted FLOPs/bytes. Compare with observed timing; identify omitted overhead rather than forcing the observation to fit the model.

**Evidence:** Derivation, assumptions, predicted-versus-measured plot, and a ranked next experiment. A bound is not a measurement.

**Expert challenge:** If your “measured bandwidth” exceeds the advertised device bandwidth, is the GPU breaking physics or is your byte-counting model wrong?

**Reading:** [Nsight Compute roofline material][NCU], [CUDA scaling guidance][CUDA-BP].

<a id="day-14"></a>
### Day 14 — Review 2 and v0.2

> **Question:** Can you teach a GPU optimization without hiding behind CUDA terminology?

**Why it matters:** Correct kernels and compelling explanations are different accomplishments. This checkpoint tests both.

**Study:** Revisit the weakest memory, synchronization, or numerical concept. No new framework today.

**Build and verify:** Reproduce one useful change and one failed or neutral change. Explain why the failure was informative. Ask another engineer to inspect a kernel, a test, or the benchmark boundary. Test at least one unfamiliar input shape without relying on the earlier solution.

**Evidence:** `v0.2` after its gate passes; Week-2 article; test summary; profiler artifacts or disclosed limitations; prioritized review findings.

**Expert challenge:** Which of your claims survives when the input shape changes—and which was specific to one convenient benchmark?

**Reading:** Own kernels and traces, checked against [CUDA references][CUDA-GUIDE].

---

### Phase 3 — From one request to a measured serving system

**Days 15–21:** Turn the Day-1 mental model into a controlled multi-user inference study. Keep the model fixed while examining queueing, cache capacity, scheduling, precision, and launch overhead.

<a id="day-15"></a>
### Day 15 — Inference anatomy and serving baseline

> **Question:** Why can a fast model forward still feel slow to a user?

**Why it matters:** The Day-1 timer measured synchronized model forwards. A user also experiences request handling, queueing, tokenization, output delivery, and other overhead. These need separate measurement boundaries.

**Study:** Streaming; request arrival; prefill/decode; client-observed time to first token (TTFT); output-token spacing; end-to-end latency; throughput; goodput.

**Build and measure:** Deploy one pinned, compatible vLLM environment on a private endpoint. Use the model’s chat template for realistic requests. Define short-input/short-output, long-input/short-output, and short-input/long-output workloads. Start with one client, then two identifiable requests; preserve arrival and completion data.

**Evidence:** Serving manifest, workload definitions, raw client records, metric definitions, and errors. Do not present changing from Transformers to vLLM as a single-kernel optimization.

**Expert challenge:** Does the client receive individual tokens or text chunks, and how does that affect your latency measurement?

**Reading:** [vLLM benchmark metrics][VLLM-BENCH], [chat templates][HF-CHAT].

<a id="day-16"></a>
### Day 16 — KV cache and capacity

> **Question:** Why do longer conversations need more memory even though the model weights are unchanged?

**Why it matters:** Multi-user capacity depends on live request state, not just weight storage. We need to connect model configuration, context lengths, and serving-engine allocation policies.

**Study:** Layers, KV heads, head dimension, precision, live sequences, logical cache contents, reservation, and allocation overhead. Query-head count and KV-head count need not be the same.

**Build and compare:** For a conventional full-attention KV cache, start with `2 × layers × KV_heads × head_dim × bytes_per_element × sum(live_sequence_lengths)`. State that this excludes weights, padding/block overhead, temporary buffers, and architecture-specific exceptions. Compare estimates with a small feasible context/concurrency sweep and observed admission behavior.

**Evidence:** Actual model fields, units, predicted memory, measured allocation/capacity, and explained discrepancies.

**Expert challenge:** Why might a serving engine reserve similar GPU memory for one and several active requests?

**Reading:** [KV-cache mechanics][HF-CACHE], [vLLM memory and scheduling guidance][VLLM-TUNING].

<a id="day-17"></a>
### Day 17 — Scheduling and batching

> **Question:** Can a long prompt delay another user whose request is tiny?

**Why it matters:** We now connect the Day-5 queue to real inference. Sharing model weights does not eliminate competition for compute, cache capacity, or scheduling time.

**Study:** Sequential service, static batching, continuous scheduling, padding, head-of-line blocking, fairness, offered load, completed throughput, and tail latency. Continuous batching schedules work; it does not make every request execute independently at the same instant.

**Build and measure:** Compare a controlled two-user scenario under sequential/static-batch baselines and the serving engine. Keep model, token counts, generation settings, and arrivals comparable. Then vary one supported scheduler setting. Expand to 2/4/8 concurrent clients only as capacity allows; do not jump blindly to 50.

**Evidence:** Request timelines, latency/throughput/error records, and a chosen operating point with its trade-off.

**Expert challenge:** Did throughput improve because scheduling improved, or because the workload silently produced fewer output tokens?

**Reading:** [vLLM tuning][VLLM-TUNING], [benchmark load controls][VLLM-BENCH].

<a id="day-18"></a>
### Day 18 — Prefix caching

> **Question:** When should previously computed prompt state be reused, and when should it not?

**Why it matters:** A repeated system prompt may offer reusable work, but an all-cache-hit demo is not representative of every workload. Shared reuse also raises tenant-boundary questions.

**Study:** Prefix matching, cache keys, hit/miss conditions, eviction, reset procedure, cache isolation, and the difference between KV-prefix reuse and returning a cached answer.

**Build and measure:** Construct explicit hit and miss workloads with matched lengths and generation settings. Record cache preparation separately from measurement. Repeat with a defined mixed hit rate. Inspect supported tenant isolation/salting behavior rather than assuming request history may be shared freely.

**Evidence:** Hit/miss protocol, raw measurements, workload assumptions, and a documented isolation test or unresolved risk.

**Expert challenge:** How could a benchmark make prefix caching look extraordinary while providing little benefit for actual users?

**Reading:** [vLLM automatic prefix caching][VLLM-PREFIX].

<a id="day-19"></a>
### Day 19 — Quantization and quality

> **Question:** Does using fewer bits automatically improve useful performance?

**Why it matters:** A memory-saving representation is valuable only when supported execution kernels and acceptable output quality make it useful for the workload.

**Study:** Weight versus activation precision, scales and metadata, quantization overhead, hardware support, accumulation behavior, and quality evaluation limits.

**Build and compare:** Choose one method supported by the actual RTX A6000 and selected runtime version. Compare with the baseline using a fixed evaluation set, explicit quality criterion, identical load, and actual output lengths. If only a separately prepared checkpoint is available, document its provenance as another variable. If unsupported, record the constraint rather than silently offloading.

**Evidence:** Compatibility record, quality checks, memory/latency/goodput comparison, failures, and a narrowly stated recommendation.

**Expert challenge:** Would you accept higher throughput if task accuracy fell below the service’s requirement?

**Reading:** [vLLM quantization compatibility][VLLM-QUANTIZATION] and the chosen method’s own documentation.

<a id="day-20"></a>
### Day 20 — CUDA Graphs and launch overhead

> **Question:** When does reducing CPU-side launch work help—and when does it barely matter?

**Why it matters:** Some workloads spend meaningful time coordinating many small GPU operations. Others are dominated by larger compute or data-movement costs; a graph is not a universal speedup.

**Study:** Capture versus replay, supported operations, shape and address constraints, graph memory, warm-up, and dispatch overhead.

**Build and measure:** Use one supported graph/no-graph runtime configuration or a standalone microbenchmark. Match shapes, numerical behavior, and warm-up; report capture cost separately from steady-state replay. Compare unprofiled timings and use traces to investigate the mechanism.

**Evidence:** Correctness check, configuration, repeated comparison, startup/steady-state distinction, and a workload-bounded conclusion.

**Expert challenge:** Did capture improve the steady-state case while making cold-start behavior or memory use worse?

**Reading:** [CUDA graph programming][CUDA-GUIDE], [PyTorch CUDA Graphs][PYTORCH-CUDA].

<a id="day-21"></a>
### Day 21 — Review 3 and v0.3

> **Question:** Which optimization claims still hold when another engineer reruns the experiment?

**Why it matters:** Combining individually promising settings can create different behavior. The final configuration needs its own comparison rather than a multiplication of separate speedups.

**Study:** Repeatability, sample size, tails, interacting changes, quality, failure accounting, and goodput.

**Build and verify:** Select only justified changes. Repeat baseline-versus-candidate tests with the same workload and recorded environment. Include a configuration that did not help, or explain why none was retained. Check raw results against figures and summaries.

**Evidence:** `v0.3` after its gate passes; Week-3 case study; raw data and chart-generation code; sample counts; limitations; outside critique request.

**Expert challenge:** Would your conclusion survive a changed prompt-length distribution, or is it specific to one controlled case?

**Reading:** Own reports and [vLLM metric definitions][VLLM-BENCH].

---

### Phase 4 — Make the system reproducible, bounded, and explainable

**Days 22–28:** Deepen reliability and security, recover the environment, reason about distributed execution, and connect measured behavior to capacity and hardware choices.

<a id="day-22"></a>
### Day 22 — Reliability, overload and security

> **Question:** What should the system do when requests exceed safe capacity or violate its assumptions?

**Why it matters:** Fast happy-path responses do not establish a usable service. Failure behavior, tenant isolation, and resource bounds are part of the design.

**Study:** Validation, admission control, bounded waiting, deadlines, cancellation, rate limits, authentication, sensitive logging, health/readiness checks, and recovery.

**Build and test:** Exercise invalid inputs, oversized requests, disconnects, overload, and a model-process failure in the owned lab. Verify the handling of cancelled work and cache cleanup. Use synthetic data and request IDs; do not publish prompt contents or tokens. Keep the endpoint private until access controls are verified.

**Evidence:** Expected-versus-actual failure matrix, recovery observations, one actionable alert, threat model, and remaining risks.

**Expert challenge:** Can a single authenticated user still exhaust shared capacity? What bounds limit the damage?

**Reading:** Installed serving-engine documentation and the project’s threat/failure model.

<a id="day-23"></a>
### Day 23 — Reproducible deployment

> **Question:** If the spot machine disappears, what survives and how quickly can useful work resume?

**Why it matters:** An image recreates software; preserved state restores work. Neither automatically resumes a running Python process or its GPU memory.

**Study:** Host versus container dependencies, immutable image digests, lock files, model revisions, mounts, backups, startup checks, readiness, and restart versus application checkpointing.

**Build and verify:** Rebuild from the documented environment, restore code/results from a separate copy, and run smoke tests. Preserve original artifacts before testing recovery. Handle package-manager locks safely. Capture exact versions and credential requirements without embedding secrets. Add Kubernetes manifests only if the single-node path is already reproducible.

**Evidence:** Clean-run transcript, time-to-readiness measurement, manifest, tested recovery procedure, and documented data-loss boundaries.

**Expert challenge:** Could the same Docker tag produce a different environment next week, and how would your record detect it?

**Reading:** [Docker build practices][DOCKER-BUILD], [NVIDIA container architecture][NVIDIA-CONTAINERS].

<a id="day-24"></a>
### Day 24 — Distributed communication — gated

> **Question:** Why can adding another GPU add waiting rather than proportional speed?

**Why it matters:** Distributed execution introduces coordination and data movement. Two independent replicas and one model split across devices are different designs.

**Study:** Rank/world size; collective operations such as all-reduce and all-gather; topology; message size; latency/bandwidth; communication correctness.

**Build and measure:** Only after approving the cost and confirming actual multi-GPU access, run a small collective test and validate its output. Record the device interconnect and message sizes. A single-GPU VM does not become a multi-GPU system because a diagram contains two boxes.

**Evidence:** Topology and measured collective results **or** a clearly labeled design-only note with a future test protocol. Either is an honest outcome; they demonstrate different depth.

**Expert challenge:** For a model that already fits on one GPU, when would replicas be preferable to splitting it?

**Reading:** [NCCL documentation][NCCL].

<a id="day-25"></a>
### Day 25 — Networking and scale-out reasoning

> **Question:** Which communication belongs inside a machine, and which crosses the network?

**Why it matters:** “Faster networking” is not a diagnosis. We need to identify what bytes move, how often, and whether computation can proceed while they move.

**Study:** PCIe, NVLink/NVSwitch as platform-dependent technologies, network interfaces, RDMA concepts, locality, communication overlap, and tensor/pipeline/data-parallel trade-offs.

**Build and reason:** Map the actual available topology. Estimate transfer costs with an explicitly simplified latency-plus-payload/bandwidth model. Compare to measurements only where hardware access exists. Identify which architecture features the present cloud flavor does not expose; never assume RDMA or NVLink from the GPU’s product name.

**Evidence:** Topology note, communication map, sourced assumptions, measured-versus-estimated values, and likely scaling limits.

**Expert challenge:** Which collective or data dependency would remain on the critical path even with a faster link?

**Reading:** [NCCL concepts and usage][NCCL].

<a id="day-26"></a>
### Day 26 — Capacity and economics

> **Question:** How many replicas are needed to meet a defined workload and latency objective at a defensible cost?

**Why it matters:** Peak token throughput is not automatically useful capacity. We need completed work that also meets response-time, correctness, and reliability requirements.

**Study:** Service-level objectives (SLOs), goodput, utilization, queueing, headroom, failures, cold starts, and cost per useful output.

**Build and analyze:** Use measured single-node results to propose capacity for an explicit workload. Add headroom and sensitivity cases rather than assuming linear scaling. Record compute, retained storage, startup/idle time, and network charges where applicable. Keep cost inputs editable, dated, and sourced from the actual billing context.

**Evidence:** Capacity/cost report, assumptions, confidence limits, and the next load test required to validate the estimate.

**Expert challenge:** Can the cheaper hourly GPU cost more per successful request once latency and idle time are included?

**Reading:** Own measurements and [goodput/load definitions][VLLM-BENCH].

<a id="day-27"></a>
### Day 27 — Hardware–software co-design

> **Question:** After finding a bottleneck, should we change the workload, software, or hardware?

**Why it matters:** Strong architecture reasoning connects an observed constraint to alternatives. Buying more compute does not necessarily address a memory, dispatch, or communication problem.

**Study:** Resource balance, cross-layer changes, counterfactual estimates, new bottlenecks, portability, and limits of analytical models.

**Build and reason:** Write a two-page memo around one measured finding. Compare a software change with hypothetical bandwidth, compute, cache, or topology changes. State what remains unverified and how it would be tested. As an optional reading exercise, map CUDA concepts to HIP/ROCm without claiming an AMD implementation or benchmark.

**Evidence:** Problem, evidence, alternatives, assumptions, counterarguments, and a recommended next experiment—not an invented silicon-simulation result.

**Expert challenge:** If the current bottleneck vanished, which resource would limit the system next?

**Reading:** [Roofline analysis][NCU], [AMD HIP documentation][HIP].

<a id="day-28"></a>
### Day 28 — Review 4 and upstream contribution

> **Question:** Can this work help another engineer, not just demonstrate personal learning?

**Why it matters:** Independent review reveals blind spots. A useful upstream contribution requires understanding someone else’s system and respecting its standards.

**Study:** Reproducible issue reports, contribution guidelines, scoped fixes, test evidence, attribution, and review responses.

**Build and review:** Consolidate the production-oriented notes. Request focused feedback on one artifact. Reproduce a genuine documentation, benchmark, or code issue in a relevant open-source project; submit only a useful minimal contribution when justified. This is a lighter review day, not a rushed hunt for a cosmetic pull request.

**Evidence:** Week-4 note, review request/response, and an issue or PR link only if actually created. Acceptance or merging is not assumed.

**Expert challenge:** What would a maintainer need to reproduce the problem without your cloud account?

**Reading:** The selected project’s own contribution guide.

---

### Phase 5 — Defend and release the work

<a id="day-29"></a>
### Day 29 — Mock loop and public explanation

> **Question:** Can you reason through an unfamiliar failure instead of replaying your own demo?

**Why it matters:** Technical credibility includes transferring understanding to new cases. Interview practice is useful when it tests this ability rather than rehearsed terminology.

**Study:** Independent implementation, diagnosis, architecture explanation, and truthful ownership/leadership stories drawn from real experience.

**Build and assess:** Run a coding, performance-diagnosis, and system-design mock. Ask the reviewer to change one assumption: shape, memory limit, request mix, failure mode, or topology. Record where hints were required. Finish an evidence-based article and a 15–20-minute talk draft.

**Evidence:** Feedback, three priority gaps, one independently repeated task, and a candidate packet linked to actual artifacts. A new lab does not substitute for unearned production leadership.

**Expert challenge:** Which layer would you inspect first, and what result would make you change direction?

**Reading:** Own project and reviewer feedback.

<a id="day-30"></a>
### Day 30 — Final release and next cycle

> **Question:** Can you defend the path from request to silicon, including what you have not built?

**Why it matters:** A release is a reproducible body of evidence, not a completion badge awarded by the calendar.

**Build and verify:** Re-run the key comparison from documented instructions. Check tests, raw data, figures, manifest, security notes, licensing/attribution, and limitations. Publish only supported findings. Label unfinished or design-only parts clearly; if the release gate is not met, publish a progress release rather than rename it production-ready.

**Evidence:** `v1.0` when justified, final article/talk links when published, retrospective, peer-review status, and one next-cycle priority.

**Expert challenge:** What is your strongest demonstrated skill, weakest assumption, and most valuable next experiment?

**Final presentation:** Explain one real improvement or a well-supported decision not to optimize, then show the evidence that could convince a skeptical engineer.

---

<a id="evidence-standard"></a>
## 6. Benchmark and evidence standards

### Before trusting a performance number

| Check | Required practice |
|---|---|
| Correctness | Compare against a defined reference; state tolerances, generation settings, and quality limitations. |
| Measurement boundary | Distinguish kernel time, synchronized model-forward time, and client-observed service latency. |
| Warm-up | Warm every measured path, including decode; report cold-start behavior separately. |
| Memory lifetime | Release all relevant references intentionally. `_` is a Python variable, not an automatic disposal mechanism. |
| Allocator state | Do not repeatedly clear the allocator cache in a steady-state loop; label deliberate cold-allocation tests separately. |
| Experimental control | Hold model revision, hardware, precision, backend, workload, and timing method fixed for a single-variable comparison. |
| Load control | Record offered request rate and concurrency separately, along with actual arrivals, completions, failures, and output token counts. |
| Sampling | Retain raw repetitions, disclose sample counts, and avoid strong tail claims from tiny samples. |
| Profiling | Record profiler settings and permissions; distinguish profiled timings from normal execution. |
| Attribution | A kernel microbenchmark gain is not an application gain unless integration and end-to-end results demonstrate it. |

The Day-1 figures remain useful exploratory observations. They are **not** end-to-end TTFT/TPOT, an established multi-user capacity, or proof that decode is context-independent. The [benchmark audit][DAY01-NOTES] explains the original script’s limitations. PyTorch’s [CUDA semantics][PYTORCH-CUDA] and vLLM’s [benchmark definitions][VLLM-BENCH] provide the relevant implementation references.

### Completion gate for any day

- [ ] The question and prediction are written down.
- [ ] The experiment or design has a clearly defined scope.
- [ ] Correctness or internal consistency has been checked.
- [ ] Raw results and the environment are recorded where measurement applies.
- [ ] The explanation identifies what is observed versus still hypothesized.
- [ ] At least one limitation or alternative explanation is documented.
- [ ] I can explain the important code and trade-off without generated assistance.
- [ ] The artifact exists; a filename in the syllabus is not evidence of completion.

For a design-only day, use assumptions and a validation plan—not invented timings—to satisfy the evidence requirement.

### Daily lab-note template

```markdown
# Day XX — Topic

## Question and why it matters
## Prediction and what would falsify it
## Relevant first principles
## Environment and exact commands
## Experiment and correctness checks
## Raw observations
## Interpretation and alternative explanations
## Failure / debugging casebook
## Limitations
## What changed in my mental model?
## One next experiment
## References and evidence links
```

---

<a id="publication"></a>
## 7. Repository and publication workflow

### Proposed repository layout

Place this file at the repository root as `SYLLABUS.md`. The Day-1 links assume the following placement. The tree is a target organization, not a claim that every file already exists.

```text
ai-systems-performance-lab/
├── README.md
├── SYLLABUS.md
├── .gitignore
├── environment/             # Manifests; no credentials
├── docker/                  # Dockerfiles and dependency definitions
├── labs/
│   ├── day01-baseline-execution-map/
│   │   ├── DAY01_LAB_NOTES.md
│   │   ├── DAY01_CURIOSITY_QUESTIONS.md
│   │   ├── 01_tokenizer.py
│   │   ├── 02_load_model.py
│   │   ├── 03_inference.py
│   │   ├── 04_prefill_decode.py
│   │   └── 05_context_benchmark.py
│   └── dayXX-topic/          # Add as each lab is implemented
├── src/                     # Reusable project components
├── tests/                   # Correctness, smoke and failure tests
├── benchmarks/              # Versioned workloads and measurement code
├── results/                 # Small sanitized data; larger artifact references
├── diagrams/
└── articles/
```

**Publication rules:** Keep code, data, and interpretation connected. Preserve the original benchmark before committing its correction. Keep model weights, caches, private keys, credentials, and sensitive logs out of Git. Attribute reused code and respect model/dependency licenses. CPU tests can run in lightweight CI; GPU tests must be explicitly run on suitable hardware or marked skipped—not passed by assumption.

### A small publication sequence

| Checkpoint | Suggested article question |
|---|---|
| Week 1 | What happens between a shell command and a GPU-generated token? |
| Week 2 | How do memory access and reuse change GPU performance? |
| Week 3 | Why do latency, throughput, cache capacity, and multi-user scheduling conflict? |
| Week 4 | What separates a working inference demo from a recoverable, bounded service? |
| Final | From token request to GPU: what I measured, what I corrected, and what remains unknown. |

Draft from lab notes rather than creating a separate content project. Seek one specific critique each week, such as “Which conclusion is unsupported?” A review request is not the same as an expert endorsement.

---

<a id="resources"></a>
## 8. Resources and hardware budget

### Use books as references, not completion targets

| Resource | Use it for |
|---|---|
| **Computer Systems: A Programmer’s Perspective** — Bryant and O’Hallaron | Days 2–5: execution, memory, linking, and concurrency. [Author site][CSAPP]. |
| **Programming Massively Parallel Processors** — Hwu, Kirk, and El Hajj | Days 6–11: parallel decomposition, memory access, reduction, and tiling. Use the existing/library edition. |
| **AI Systems Performance Engineering** — Chris Fregly | Profiling methodology and the connection between kernels, inference, and infrastructure. [Author’s companion repository][FREGLY]. |
| Previously completed inference reading | Retrieval practice: explain prefill, KV cache, and serving trade-offs, then test them. Do not restart a whole book unnecessarily. |

### Primary documentation

| Area | Working reference |
|---|---|
| OS visibility | [Linux `/proc`][LINUX-PROC] |
| C++ engineering | [C++ Core Guidelines][CPP-GUIDELINES] |
| CUDA programming | [CUDA 12.8 Programming Guide][CUDA-GUIDE] |
| CUDA methodology | [CUDA Best Practices][CUDA-BP] |
| Numerical and synchronization checking | [Compute Sanitizer][COMPUTE-SANITIZER] |
| GPU container boundary | [NVIDIA Container Toolkit architecture][NVIDIA-CONTAINERS] |
| Tensor execution and memory | [PyTorch 2.7 CUDA semantics][PYTORCH-CUDA] |
| Model cache and input format | [Hugging Face cache explanation][HF-CACHE] and [chat templates][HF-CHAT] |
| Timeline diagnosis | [Nsight Systems][NSYS] |
| Kernel diagnosis and roofline | [Nsight Compute][NCU] |
| Serving measurement | [vLLM benchmark CLI][VLLM-BENCH] |
| Scheduling and capacity | [vLLM optimization guidance][VLLM-TUNING] |
| Cache/precision experiments | [Prefix caching][VLLM-PREFIX] and [quantization support][VLLM-QUANTIZATION] |
| Rebuilds | [Docker build practices][DOCKER-BUILD] |
| Distributed execution | [NCCL][NCCL] |
| Later cross-vendor translation | [HIP][HIP] |

**Version rule:** References with `latest` or otherwise moving content can differ from the installed software. Use the corresponding release documentation and source revision for actual commands. PyTorch 2.7 and CUDA 12.8 links preserve the initial lab’s reference point; they are not a demand that every future environment use those versions.

### Spend on the current question

Reuse books and library access. Start with the available single-GPU lab. Days focused on C++/Linux, writing, or analysis do not require keeping an expensive GPU running continuously. Preserve state and verify provider billing behavior before releasing resources.

Track compute allocation time, persistent-storage charges, model-download overhead, and review costs separately. Do not reuse historical spot prices as current quotes. Approve multi-GPU spending only after defining the Day-24 experiment and its acceptance criteria. A private cloud session and a backup outside that session serve different purposes.

Do not buy another course or GPU simply because an unfamiliar term appears. Find the smallest experiment that answers the question first.

---

<a id="next-cycle"></a>
## 9. Final assessment and the next year

### The month’s evidence package

The release should make it straightforward for an engineer to find:

1. The question, assumptions, and execution map.
2. Source, environment definitions, correctness checks, and reproduction commands.
3. Raw measurements, profiling evidence, and a bounded performance explanation.
4. Multi-user and failure behavior that was actually tested.
5. Capacity/cost assumptions and an explicit list of unimplemented extensions.
6. A readable retrospective showing mistakes, corrections, and the next knowledge gap.

### Self-assessment

Score each skill separately for **explanation, implementation, testing, measurement, and defense of trade-offs**:

| Score | Meaning |
|---|---|
| 0 | Not yet demonstrated |
| 1 | Can proceed with substantial prompts or assistance |
| 2 | Can work independently on the familiar lab |
| 3 | Can transfer the reasoning to an unfamiliar variation |

These are internal learning checks, not hiring-level certifications. Relevant role families include GPU software, inference performance, ML systems, accelerated infrastructure, and performance modeling. A project strengthens evidence; it does not guarantee a title, compensation, or production track record.

### One-year continuation: deepen, do not multiply missions

| Period | One direction to deepen | Evidence target |
|---|---|---|
| Months 2–3 | The weakest foundation revealed by the sprint: C++/CUDA, profiling, or inference runtime internals | Independent implementation, more rigorous experiments, meaningful review |
| Months 4–6 | Production operation and distributed execution on hardware actually available | Recovery/load tests, real topology measurements, a useful upstream contribution |
| Months 7–9 | One specialization: kernels, serving/scheduling, or performance modeling | A substantial investigation that survives external technical critique |
| Months 10–12 | Lead a real systems improvement and teach the reasoning | Documented ownership, measured impact, technical writing/talks, and mentoring |

Compiler internals, deeper Triton/CUTLASS work, AMD implementation, speculative decoding, disaggregated serving, and fleet-scale Kubernetes remain possible follow-up topics. They enter the active plan only when the evidence identifies a reason—not because the list is interesting.

> **Finish the current evidence package. Keep curiosity wide, but active execution narrow.**

---

## Reference links

Internal links below assume this file is uploaded at the repository root with the Day-1 files in the documented directory. Technical links are references, not statements that every feature is supported by the lab’s installed versions.

[DAY01-NOTES]: labs/day01-baseline-execution-map/DAY01_LAB_NOTES.md
[DAY01-QUESTIONS]: labs/day01-baseline-execution-map/DAY01_CURIOSITY_QUESTIONS.md
[CSAPP]: https://csapp.cs.cmu.edu/3e/home.html
[LINUX-PROC]: https://docs.kernel.org/filesystems/proc.html
[CPP-GUIDELINES]: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
[CUDA-GUIDE]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html
[CUDA-BP]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
[COMPUTE-SANITIZER]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
[NVIDIA-CONTAINERS]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html
[PYTORCH-CUDA]: https://docs.pytorch.org/docs/2.7/notes/cuda.html
[HF-CACHE]: https://huggingface.co/docs/transformers/cache_explanation
[HF-CHAT]: https://huggingface.co/docs/transformers/chat_templating
[NSYS]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
[NCU]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
[VLLM-BENCH]: https://docs.vllm.ai/en/latest/cli/bench/serve/
[VLLM-TUNING]: https://docs.vllm.ai/en/latest/configuration/optimization/
[VLLM-PREFIX]: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
[VLLM-QUANTIZATION]: https://docs.vllm.ai/en/latest/features/quantization/
[DOCKER-BUILD]: https://docs.docker.com/build/building/best-practices/
[NCCL]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
[HIP]: https://rocm.docs.amd.com/projects/HIP/en/latest/
[FREGLY]: https://github.com/cfregly/ai-performance-engineering
