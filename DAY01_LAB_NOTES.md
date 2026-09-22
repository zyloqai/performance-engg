# Day 01 — From a Cloud VM to Transformer Prefill and Decode

**Project:** AI Systems Performance Lab  
**Lab date:** 21 September 2026  
**Lab owner:** Sankalp  
**Status:** Completed learning experiments; exploratory measurements; benchmark revision required  
**Companion:** [Day-1 curiosity and expert-review workbook](DAY01_CURIOSITY_QUESTIONS.md)

> **Starting question:** What actually happens between submitting a text prompt and computing the next token on a GPU?
>
> **Learning method:** Question → prediction → small experiment → observation → explanation → challenge the explanation.

This is a reconstructed engineering record of the work performed during Day 1. Measurements below come from the terminal outputs shared during the lab, not from a new run conducted while writing this document. Provider addresses, key fingerprints, instance IDs and account details have been omitted.

**Evidence labels used throughout:**

- **Observed:** present in the shared terminal output or screenshot.
- **Reported complete:** the learner confirmed completion, but the detailed output was not retained here.
- **Derived:** calculated from stated assumptions or recorded observations.
- **Proposed:** a follow-up experiment, not a result already obtained.

This is not a claim of production readiness, a GPU ranking, or an enterprise throughput benchmark. It is evidence of a working execution path and the beginning of a disciplined performance investigation.

## Contents

1. [What Day 1 achieved](#1-what-day-1-achieved)
2. [Environment and remaining unknowns](#2-environment-and-remaining-unknowns)
3. [Three maps, not one misleading stack](#3-three-maps-not-one-misleading-stack)
4. [Provisioning decisions and resource accounting](#4-provisioning-decisions-and-resource-accounting)
5. [Debugging casebook](#5-debugging-casebook)
6. [Containers, processes and persistent work](#6-containers-processes-and-persistent-work)
7. [Experiments from tensors to tokens](#7-experiments-from-tensors-to-tokens)
8. [Context-length benchmark and raw observations](#8-context-length-benchmark-and-raw-observations)
9. [Audit of the original benchmark](#9-audit-of-the-original-benchmark)
10. [First-principles memory estimates](#10-first-principles-memory-estimates)
11. [Capture and publish the evidence](#11-capture-and-publish-the-evidence)
12. [What is complete and what comes next](#12-what-is-complete-and-what-comes-next)
13. [Sources](#13-sources)

---

## 1. What Day 1 achieved

We moved through progressively stronger tests:

| Milestone | Evidence | What it does **not** establish |
|---|---|---|
| Connect to the cloud machine | TCP/22 eventually succeeded; SSH login subsequently worked | GPU health or application readiness |
| Inspect GPU from the host | NVIDIA driver and GPU could be queried | CUDA Toolkit installation or working inference |
| Run an ordinary container | `hello-world` printed successfully | Container GPU access |
| Expose GPU to a container | The previously failing `--gpus all` command returned GPU status | Correct model output or sustained compute performance |
| Execute a GPU tensor operation | `[1, 2, 3] × 10` produced `[10, 20, 30]` on `cuda:0` | Kernel efficiency |
| Predict a large allocation | A 4-GiB tensor exercise was reported complete | An archived allocator trace or exact observed delta |
| Tokenize a sentence | Nine token IDs in a CPU tensor | GPU execution |
| Execute model prefill and cached decode | Two generated tokens and separate forward-pass timings | End-to-end user latency |
| Vary context length | Five measurements at each of three lengths | A production benchmark or a proven bottleneck |

The progression matters: each check answered a different question. A successful lower-layer check was not treated as proof that every layer above it worked.

## 2. Environment and remaining unknowns

### 2.1 Recorded configuration

| Item | Value | Evidence status |
|---|---|---|
| Cloud | AceCloud spot VM | Observed provisioning workflow |
| Final GPU | NVIDIA RTX A6000 | Observed in `nvidia-smi` and PyTorch |
| GPU family | Ampere; nominal 48-GB GDDR6 product | NVIDIA specification, not a measurement [S01] |
| GPU memory exposed | **46,068 MiB**, approximately **44.99 GiB** | Observed in `nvidia-smi` |
| Selected VM flavor | `N.RTXA6000.64`: 16 vCPUs, 64 GB system RAM | Provider selection; guest CPU topology and usable RAM still need capture |
| Boot storage | 150 GB requested | Selected configuration; actual filesystem capacity still needs capture |
| Boot image | Ubuntu 24.04 LTS NVIDIA 580 Base | Provider image selection; capture guest release/kernel before publishing a final manifest |
| NVIDIA driver | **580.178.04** | Observed |
| `nvidia-smi` CUDA field | **13.0** | Driver-supported CUDA level; not an installed Toolkit version [S02] |
| Container image tag | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` | Used in the lab; immutable image digest not yet captured |
| Python | **3.11.13** | Observed in the container |
| PyTorch | **2.7.1+cu128** | Observed during the tensor experiment |
| `torch.version.cuda` | **12.8** | Observed; reports the CUDA version PyTorch was built against |
| CUDA available | `True` | Observed in PyTorch |
| Model | `Qwen/Qwen2.5-7B-Instruct` | Model ID used by the scripts |
| Intended weight dtype | BF16 | Explicit script setting; audit parameter and buffer dtypes before a final benchmark |
| Batch size | 1 | Explicit script setting |
| Transformers, Accelerate, Safetensors | Installed in `gpu-lab` | Exact versions not preserved in the shared output |
| Attention implementation | Not recorded | Do not retroactively claim FlashAttention or a specific SDPA kernel |
| Model/tokenizer commit | Not recorded | Model ID alone is not an immutable revision |
| Container Toolkit and Docker versions | Installation succeeded | Capture exact installed versions |

The current library inventory should be captured again. Later `pip install` operations can change dependencies; the earlier tensor-test version output is not a substitute for the final environment manifest.

**GPU naming correction:** RTX **A6000** is not RTX **6000 Ada**, and neither is RTX **PRO 6000 Blackwell**. They must not be merged into a single benchmark label. Earlier provisioning discussions included other GPUs, but the recorded inference results here belong to the **RTX A6000**.

**Memory correction:** do not explain the difference between nominal GPU capacity and 46,068 MiB using unit conversion alone. Device reservations, ECC and configuration are possible factors; their contribution was not diagnosed in this lab. Record the actual exposed capacity and investigate before assigning a cause.

### 2.2 The idle observation

The container-side `nvidia-smi` screenshot showed 0 MiB used, 0% GPU utilization, 30°C, performance state P8 and approximately 6 W against a 300-W limit. No GPU process was listed at that instant.

These are a sampled idle observation, not evidence that the machine will remain idle or that every process on the host is visible from every PID namespace. A status screen is not an execution trace. [S02]

## 3. Three maps, not one misleading stack

### 3.1 Access map: how the learner reaches the machine

```text
Windows terminal
    │ SSH connection initiated over TCP
    ▼
Cloud networking / security-group rules
    ▼
Ubuntu VM: sshd
    │ host-key checks + user authentication
    ▼
Remote shell
    │ docker exec / python commands
    ▼
Experiment process on the cloud machine
```

Typing through SSH does not make the laptop execute the model. The laptop provides input and displays terminal output; the model process runs remotely.

SSH was our **control connection**, not an HTTP inference endpoint. The benchmark did not include internet round-trip time to an API client.

### 3.2 Container setup map: how GPU access becomes available

```text
docker run --gpus all ...
    ▼
Docker / container lifecycle machinery
    ▼
NVIDIA container integration configures GPU devices and driver access
    ▼
Container starts with the necessary access available
```

NVIDIA Container Toolkit is a collection of integration components. The NVIDIA container runtime is one component; its implementation and hooks modify container configuration to make the required devices and driver components available. [S03]

**Important correction:** the Toolkit is not a GPU arithmetic engine or a separate network hop traversed for every tensor operation. Docker is not multiplying the matrices. Its role here is to create and manage the environment in which the process runs.

### 3.3 Application execution map: what computes the token

```text
Container userspace, executing on the host's CPU
    Python → Transformers → PyTorch / CUDA user-space libraries
                                  │ dispatch work
                                  ▼
                   Host NVIDIA driver and GPU interface
                                  ▼
                         RTX A6000 execution
                                  │
                                  ▼
                   GPU-resident weights, inputs and KV cache
```

The kernel driver remains part of the host kernel environment. An ordinary Linux container is not a second independently booted kernel.

There are also two different uses of the word **runtime**:

| Term | Purpose |
|---|---|
| Container runtime | Creates/runs a containerized process environment |
| CUDA runtime | Application-facing CUDA services such as memory and kernel-launch support |

The CUDA Toolkit supplies development tools such as `nvcc`. A prebuilt PyTorch runtime container can execute compiled GPU operations without a host installation of `nvcc`. If `nvcc` is not found, the immediate observation is that the executable is not on the current `PATH`; that alone does not inventory every CUDA file on disk. [S04]

### 3.4 Why other container technologies exist

These are not all competitors at the same layer:

| Technology | Role in the conceptual map | Why it is useful |
|---|---|---|
| Docker | Developer-facing engine/tooling | Build, distribute and run application environments |
| containerd | Container lifecycle management | A reusable lifecycle service beneath higher-level tools |
| runc | Low-level OCI runtime | Create the isolated process according to a runtime specification |
| Podman | Alternative engine/workflow | A different, often daemonless workflow |
| CRI-O | Kubernetes-oriented runtime service | Implement Kubernetes' container-runtime interface |
| Kubernetes | Cluster orchestration | Schedule and manage workloads across machines |
| CDI | Device-description interface | Describe device access for compatible container tooling |
| NVIDIA Container Toolkit | NVIDIA device integration | Configure NVIDIA device/library access |

This was conceptual orientation, not a popularity survey or a hands-on comparison of these tools. Day 1 used Docker and NVIDIA Container Toolkit. [S03], [S05], [S06], [S23], [S24]

## 4. Provisioning decisions and resource accounting

### 4.1 Why RTX A6000 was selected

In the final price screenshot, the L4 flavor was ₹33.34/hour and RTX A6000 was ₹37.86/hour. The difference was ₹4.52/hour, or approximately 13.6%. The selected A6000 flavor also offered more host RAM and vCPUs.

This was a **historical screenshot-based decision**, not a current price quote and not a measured cost-per-token comparison. We chose comfortable memory headroom for an unquantized baseline. We did not establish that A6000 is universally faster or cheaper per completed request.

The provider's RAM column described **system RAM**, not GPU VRAM. Disk capacity, system RAM capacity and VRAM capacity are separate resources. Enlarging one does not automatically enlarge another.

### 4.2 Image, storage and bootstrap choices

We selected a GPU-ready Ubuntu image to begin with a working driver, requested 150 GB of boot storage, and avoided inserting an untested post-creation script.

The intended architecture was:

```text
Host image: operating system + driver + Docker + GPU integration
Container image: Python + PyTorch + CUDA libraries + model dependencies
Persistent files: source + model cache + experiment results
External backup/Git: recovery and history outside the spot VM
```

A boot image is a starting filesystem/software configuration, not a snapshot of a running Python interpreter's live RAM. A golden image makes environment recreation easier; it does not automatically resume the exact in-memory execution state of an interrupted experiment.

A retained disk is also not a tested backup. Spot-reclamation behavior, volume retention, billing and any cross-region portability need explicit provider verification. We did not prove those properties here.

### 4.3 Cost discipline

Do not add another GPU merely to avoid understanding a setup error. Check that unused earlier instances and retained volumes are no longer incurring unintended charges, using the provider's actual billing rules. Termination is destructive: first verify that important data is copied and recoverable.

For later comparisons, use both:

```text
cost per experiment = billed allocation time × applicable hourly rates
cost per useful result = total cost / completed results meeting quality and latency criteria
```

The lowest price per hour is not necessarily the lowest price per useful result.

## 5. Debugging casebook

Failures were part of the experiment. The useful artifact is not only the command that eventually worked, but the evidence that led to it.

### 5.1 SSH timeout: diagnose transport before credentials

**Observed:** SSH initially timed out; `Test-NetConnection ... -Port 22` returned false.

**Reasoning:** a failed TCP connection does not test whether the private key is authorized. Investigate the address, routing, instance state, attached security groups and host/listener availability first. A failed ping is not decisive because ICMP and TCP can be filtered differently.

**Action taken:** inspect existing rules and add narrowly scoped inbound TCP/22 access from the learner's current public IPv4 address (`/32`). Outbound internet access was already present in the existing group.

**Observed afterward:** TCP/22 became reachable and the SSH error changed to public-key authentication failure. This is evidence of progress to a different layer, not a reason to repeat the same networking changes.

Reusable Windows checks, with placeholders rather than live addresses:

```powershell
Test-NetConnection <VM_PUBLIC_IP> -Port 22
ssh -o IdentitiesOnly=yes -i "$HOME\.ssh\<PRIVATE_KEY>.pem" ubuntu@<VM_PUBLIC_IP>
```

Do not put a backslash before `@`. Verify the username from the image/provider. Do not open every inbound port to make SSH work.

### 5.2 Duplicate security-group rule

**Observed:** the console reported that a rule already existed after an outbound allow rule was added to the creation form.

**Likely explanation:** the existing outbound rule was being duplicated. The message alone did not identify which submitted rule was duplicate.

**Correction:** inspect the saved rule list, preserve working rules, and add only the missing inbound rule. A form containing a group identifier may be creating rules *within* an existing group rather than creating a new group. Verify what object is being changed and which group is attached to the VM.

**Lesson:** a screen title and an error message are not a complete state inventory.

### 5.3 Public-key authentication failed after networking worked

**Observed:** SSH established a connection, exchanged keys, and the client signed an authentication request using the local RSA key. The server rejected the requested user/key combination.

**Possible causes:** wrong user, mismatched public/private key, missing key injection, file permissions or another server-side authentication policy. Client output alone does not reveal every server-side reason.

The cloud key fingerprint and the first local key's fingerprint differed after displaying them in the same MD5 style. That was consistent with a mismatched key, assuming both fingerprints used the same public-key representation and hashing convention. A key's filename is not its identity.

Useful comparison command:

```powershell
ssh-keygen -E md5 -lf "$HOME\.ssh\<PRIVATE_KEY>.pem"
```

OpenSSH supports selecting MD5 or SHA-256 for fingerprint display. A SHA-256 string must not be compared visually to an MD5 string. Use SHA-256 where the provider supports it; MD5 was used here only to compare with the provider's displayed format. [S07]

The later instance was reached successfully. We did not capture enough evidence to claim which provider-side detail caused the original provisioning mismatch, nor should we claim that changing region mathematically changed a PEM file. A region/project may expose different named key resources; the same matching public key can be registered wherever supported.

**Host key versus user key:** accepting a first-connection host key records it for future comparison. That is not the same as independently authenticating its fingerprint through a trusted provider console. User authentication is a separate direction of trust.

### 5.4 Derived public-key file was rejected

**Observed:** redirecting `ssh-keygen -y` into a `.pub` file was followed by “not a public key file.” Direct fingerprint inspection of the private-key file worked later.

**Unconfirmed explanation:** text encoding may have been involved. Windows PowerShell's redirection behavior can produce UTF-16LE output, which is inappropriate for the expected plain-text public-key format. We did not inspect the failed file's bytes, so encoding remains a hypothesis. [S08]

A less ambiguous public-key export, when needed:

```powershell
ssh-keygen -y -f "$HOME\.ssh\<PRIVATE_KEY>.pem" |
    Set-Content -Encoding ascii "$HOME\.ssh\public-key-check.pub"
ssh-keygen -E sha256 -lf "$HOME\.ssh\public-key-check.pub"
```

This exports a **public** key, not the private key. Never publish private-key material, passwords or tokens as part of an error report.

### 5.5 APT package-list lock and missing package

**Observed:** `apt update` could not obtain `/var/lib/apt/lists/lock`, held by an `apt-get` process. The subsequent Docker install could not locate its package.

**Reasoning:** the earlier update failed. Incomplete/stale indexes or repository configuration can explain a missing package; it was too strong to claim the lock was the only possible cause. Retry only after the active package operation finishes, then confirm a successful index update.

Inspect rather than destroy:

```bash
ps -fp <CURRENT_LOCK_HOLDER_PID>
cloud-init status
```

Never delete lock files as a substitute for coordinating package managers. A PID identifies a process at a point in time and can be reused; a historic PID is not a permanent program identity.

### 5.6 APT frontend lock held by unattended upgrades

**Observed:** Docker installation waited for `/var/lib/dpkg/lock-frontend`, held by an `unattended-upgr` process.

**Action:** allow the active update operation to finish. Ubuntu's unattended-upgrades mechanism can apply updates automatically. A running VM can still be performing background initialization or maintenance. [S09]

The eventual baseline must be captured **after** this activity settles. “We did not manually upgrade the driver” does not prove no background package changed during provisioning.

**Avoid:** killing package-manager processes indiscriminately, deleting locks, or permanently disabling security updates just to simplify a benchmark. Pin a tested experiment environment while maintaining an explicit patch-and-retest policy.

### 5.7 `nvidia-smi` worked but `nvcc` was absent

The driver-management path worked; the compiler executable was not available in the current shell. These are separate checks.

```text
nvidia-smi succeeds → driver-management communication works
nvcc --version succeeds → that shell can find a CUDA compiler
PyTorch tensor test succeeds → the application compute path works
```

No one of these proves all the others. `nvidia-smi` is normally supplied with driver utilities, sometimes through a separately packaged utilities component. It is not evidence that the full CUDA development toolkit is installed. [S02], [S04]

### 5.8 Ordinary Docker worked, GPU Docker failed

**Observed:** `hello-world` succeeded. Then:

```text
could not select device driver "" with capabilities: [[gpu]]
```

The host's GPU driver had already been tested. The new failure concerned Docker's ability to satisfy the GPU request.

**Change:** install NVIDIA Container Toolkit, configure Docker integration and restart Docker. Restarting Docker can affect active containers, so do this deliberately.

**Retest:** the same command succeeded afterward:

```bash
sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

NVIDIA documents this installation/configuration workflow. It does not require installing another kernel driver inside the Ubuntu container. [S10]

### 5.9 New Python process, missing import

**Observed:** after `docker exec ... bash` and starting `python`, the expression `torch.zeros(...)` raised `NameError: name 'torch' is not defined`.

**Correction:** `import torch` in that Python interpreter.

`docker exec` starts a new command in an existing container. Starting another Python interpreter does not resume the memory of the old interpreter. The old one might still exist in another session; its exit was not established by the new session. [S11]

**Lesson:** container filesystem lifetime, process lifetime and Python object lifetime are different.

### 5.10 Hugging Face authentication warning

**Observed:** public tokenizer/model requests warned about unauthenticated access, but the downloads and experiments completed.

The warning was not the cause of an inference failure. Authentication may be useful if download limits are encountered; do not embed `HF_TOKEN` into a public script, Dockerfile, shell-history example or screenshot.

### 5.11 `torch_dtype` deprecation

**Observed:** the installed Transformers version warned that `torch_dtype` was deprecated and requested `dtype` instead.

The lab adopted:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cuda",
)
```

The current model-loading documentation supports `dtype`. Record the actual installed version rather than assuming all past/future Transformers releases use identical APIs. [S12]

### 5.12 A benchmark can run successfully and still need correction

The context benchmark completed and printed plausible numbers. A subsequent code review identified allocator-state changes, incomplete warm-up and surviving object references. These are methodological problems, not evidence that the GPU failed.

The right response is to retain the exploratory run, document limitations, fix the protocol and measure again—not silently replace the old results and present the new ones as the same experiment.

## 6. Containers, processes and persistent work

### 6.1 Three lifetimes to distinguish

| Object | What survives a stop/restart? | What does not follow automatically? |
|---|---|---|
| Container image | The image can be used to start another container | Packages installed later in a particular container |
| Named container filesystem | Its writable layer remains while the container exists | Live Python memory after process/container exit |
| Host bind-mounted directory | Files remain independently of container removal | Survival after loss/deletion of the underlying host disk |

A container created with `--rm` is removed after it exits. A named container is easier to manage, but its name is not a backup strategy. Docker's overview explains the image/container distinction. [S05]

### 6.2 The lab's chosen storage layout

```text
Host: ~/ai-systems-lab/
    workspace/   → container: /workspace
    hf-cache/    → container: /root/.cache/huggingface
    results/     → container: /workspace/results
```

These are **bind mounts**: specific host directories exposed inside the container. In particular, `/workspace/results` maps to a separate host directory, not to the host's `workspace/results` contents. Back up both the code directory and the results directory. [S13]

Do not treat `/root/.cache/huggingface` as automatically safe to publish. It may contain cached credentials as well as model files. Use explicit file selection for Git.

### 6.3 Daily reconnection commands

Run these on the **Ubuntu host**, not at the Python `>>>` prompt:

```bash
sudo docker ps -a
sudo docker start gpu-lab       # only if this existing container is stopped
sudo docker exec -it gpu-lab bash
```

Inside the container, a new `python` command begins a fresh interpreter. Save experiments as `.py` files; do not depend on remembering interactive shell state.

### 6.4 Host environment versus application environment

The application packages were installed **inside** `gpu-lab`:

```bash
python -m pip install transformers accelerate safetensors
```

This modified that container's Python environment, not the original image. A later Dockerfile should reproduce a reviewed, pinned package set. An image tag alone does not capture those interactive modifications.

`--shm-size=8g` sets the container's shared-memory filesystem limit. It does not add 8 GiB of GPU VRAM and does not mean 8 GiB of physical RAM is immediately consumed. We did not establish that this particular single-process experiment needed that limit.

## 7. Experiments from tensors to tokens

### 7.1 First successful GPU arithmetic

**Observed code and results:**

```python
import torch

print(torch.__version__)                  # 2.7.1+cu128
print(torch.version.cuda)                 # 12.8
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # NVIDIA RTX A6000

x = torch.tensor([1.0, 2.0, 3.0])
print(x.device)                           # cpu
x_gpu = x.to("cuda")
print(x_gpu.device)                       # cuda:0

print(torch.cuda.memory_allocated())      # 512, observed at this point

y_gpu = x_gpu * 10
torch.cuda.synchronize()
print(y_gpu)                             # [10., 20., 30.] on cuda:0
```

The default float32 tensor payload was 3 × 4 = **12 bytes**. The observed allocator accounting was **512 bytes**, not 512 MB. Allocation granularity explains why the counter need not equal the payload; it is not evidence that float32 uses 512 bytes per number.

PyTorch's CUDA allocator rounds/manages allocations and caches released blocks. CUDA work is ordinarily asynchronous with CPU execution; synchronized boundaries are therefore important for timing. [S14]

The CPU still runs Python and dispatches operations. The GPU executes the selected GPU work. Printing a GPU tensor's numerical contents requires host-visible data; avoid such printing inside a benchmark interval.

### 7.2 Four-GiB allocation exercise

**Prediction and code:**

```python
big = torch.zeros((1024, 1024, 1024), dtype=torch.float32, device="cuda")
```

```text
1,024³ elements × 4 bytes
= 4,294,967,296 bytes
= 4 GiB
```

The learner reported completing the exercise while observing GPU memory. The exact before/after counter values were not preserved, so this document records **the prediction**, not an invented measured delta.

Three questions emerged: how much storage do tensor elements need, how much memory does PyTorch retain, and how much does the driver report? These are related but not interchangeable questions.

`del big` removes that Python reference. Other references may keep the allocation live. `empty_cache()` releases unused allocator-held memory, not still-referenced tensor storage. [S25]

### 7.3 Tokenization: text became nine CPU-side IDs

**Observed input:**

```text
Explain why GPUs are useful for AI.
```

**Observed tokens:**

```text
['Ex', 'plain', 'Ġwhy', 'ĠGPUs', 'Ġare', 'Ġuseful', 'Ġfor', 'ĠAI', '.']
```

**Observed IDs and shape:**

```text
[840, 20772, 3170, 70403, 525, 5390, 369, 15235, 13]
shape = [1, 9]
device = cpu
```

The leading `Ġ` in this tokenizer's displayed token strings represents a space. The two pieces `Ex` and `plain` demonstrate that one word need not be one token.

IDs select vocabulary/embedding entries; their numeric ordering is not a semantic scale. A token ID such as 70403 is an index, not a statement that its meaning is “larger” than token 840. The model's embedding layer converts IDs into vectors before subsequent layer computations. [S27]

A useful correction to the original motivating question: neural models *can* be built around characters or bytes. Subword tokenization is a design choice balancing vocabulary size, sequence length and representation, not a mathematical requirement that characters can never be modeled.

### 7.4 Loading weights into GPU memory

We used `AutoModelForCausalLM.from_pretrained` with BF16 and explicit CUDA placement. The initial estimate was:

```text
about 7 billion parameters × 2 bytes ≈ 14 GB decimal ≈ 13.0 GiB
```

The model's name is an approximation, not its exact parameter count. We did not retain the detailed load-only parameter-count report. Do not invent one from the name.

For the actual loaded model, calculate rather than guess:

```python
# Run where the live `model` object exists; do not load another copy just for this.
parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
print("Parameter GiB:", parameter_bytes / 2**30)
print("Buffer GiB:", buffer_bytes / 2**30)
```

This is an accounting estimate; shared storage/views and quantized representations can require additional care in a more general implementation.

Loading also involves storage/network access, CPU-side work and device transfer. We did not profile that path. A conceptual arrow “disk → RAM → GPU” must not be presented as proof that the entire model existed as one complete extra CPU copy at a particular instant.

### 7.5 Prefill and the first next-token decision

The script processed all nine prompt tokens with `use_cache=True`, then selected:

```python
next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
```

**Logits are unnormalized scores**, not probabilities. Greedy selection can use `argmax` directly. The final prompt position's scores are used to choose the first generated token. A forward pass exposing every position's logits has shape `[batch, prompt_length, vocabulary_size]`. [S15]

**Observed one-shot result:** prefill forward **575.84 ms**, selected ID **70403**, decoded text **`' GPUs'`**.

This was raw-text continuation with an instruction-tuned model. We did not apply the model's chat template, so the unusual continuation is not a quality evaluation. Chat formatting adds role/control tokens and changes the actual input length. [S16]

### 7.6 One cached decode step

The next forward pass received **only the first generated token**, an extended attention mask and the previously returned KV cache.

**Observed:** decode forward **57.98 ms**; selected ID **11**, decoded text **`','`**.

The cache contains per-layer keys and values for tokens already processed. It avoids recomputing those past token representations, but the current attention computation still uses cached context. Cache reuse is not independence from context length. [S17]

Track the sequence position carefully:

```text
Prefill consumes prompt tokens 1…9
    → cache covers 9 tokens
    → logits select output token 1

Decode consumes output token 1
    → cache covers 10 tokens
    → logits select output token 2
```

Output token 2 is not yet represented in the cache merely because it was selected. It enters the cache when it is fed into the next forward pass.

Also, the first selected token comes from prefill's final-position logits. There is not a mandatory extra decode forward before token 1 can be selected.

## 8. Context-length benchmark and raw observations

### 8.1 Experimental design

**Independent variable:** prompt length, 128 → 512 → 2,048 tokens.

**Held constant in the shared script:** model ID, intended BF16 precision, batch size 1, one GPU, synthetic token ID 1000 repeated to the desired length, two prefill warm-ups per length, five measured prefill/decode pairs and one cached decode step per pair.

The tokenizer was loaded but did not construct natural-language prompts for this sweep. These were synthetic length-controlled sequences. No multi-user HTTP server was involved.

### 8.2 Timing boundary

The helper timed this interval with a CPU clock:

```python
torch.cuda.synchronize()       # drain prior GPU work before starting
start = time.perf_counter()
with torch.inference_mode():
    out = model(**kwargs)
torch.cuda.synchronize()       # wait for submitted work to finish
elapsed_ms = (time.perf_counter() - start) * 1000
```

The metric is **synchronized model-forward wall time**. It includes Python/framework dispatch and completion waiting. It is not an individual kernel timer.

It excludes model download/load, tokenization, preparation of inputs before the helper, token selection after the helper, detokenization, request queueing and response delivery. CUDA event timing would answer a related but different timing question. [S28]

Do not rename these columns to TTFT or TPOT. Client-observed TTFT starts at request submission and ends when the client receives its first token. A sustained decode/TPOT estimate requires a clearly defined multi-token interval, not merely one isolated decode call. [S18]

### 8.3 Summary reproduced from the observed output

| Prompt tokens | Median prefill forward (ms) | Median one-token cached decode forward (ms) | Post-loop `memory_allocated()` (GiB) |
|---:|---:|---:|---:|
| 128 | 36.30 | 29.56 | 14.20 |
| 512 | 86.60 | 29.91 | 14.22 |
| 2,048 | 304.57 | 30.32 | 14.30 |

### 8.4 Raw run values

These are transcribed from the shared terminal screenshot. They are not a new benchmark execution.

```csv
prompt_tokens,run,prefill_forward_ms,decode_one_forward_ms
128,1,36.41,60.07
128,2,36.30,29.36
128,3,35.85,29.56
128,4,42.51,36.40
128,5,35.78,29.12
512,1,87.23,32.71
512,2,86.60,29.43
512,3,86.83,29.91
512,4,85.93,29.91
512,5,86.06,29.81
2048,1,307.89,30.32
2048,2,302.53,30.38
2048,3,304.19,30.58
2048,4,304.57,30.16
2048,5,304.64,30.15
```

The medians above can be recomputed directly from these values. The 60.07-ms first decode observation remains part of the record; do not delete it simply because it is inconvenient.

### 8.5 Interpretation with appropriate limits

**Observation 1:** a 16× prompt-length increase corresponded to approximately **8.39×** median prefill forward time in this run.

**Observation 2:** median cached one-token decode forward time changed from 29.56 to 30.32 ms—approximately **2.57%** over this range.

**Observation 3:** the reported post-loop allocated memory rose by approximately 0.10 GiB.

The data supports a narrow statement: *prefill forward time grew substantially across the tested lengths, while one-token cached decode forward time was nearly flat in this small sample.*

It does **not** prove any of the following:

- Decode is independent of context length.
- The GPU was definitely compute-bound or memory-bandwidth-bound.
- Prefill has a particular complexity exponent inferred from three points.
- KV cache caused a measured speedup relative to no cache—we did not run that control.
- A6000 can serve a given number of simultaneous users within an SLO.
- The first un-warmed 575.84-ms result and the later 36.30-ms result constitute an optimization speedup. Prompt length, warm-up and experimental conditions differ.

**Hypotheses to test:** shared per-step work, weight traffic and launch/dispatch overhead may dominate short cached decode; larger contexts could expose attention/cache costs more clearly. A profiler and a corrected benchmark are needed to distinguish these explanations.

## 9. Audit of the original benchmark

This review applies to the script shared during the lab. The actual VM copy and its eventual Git commit should be preserved before editing.

### 9.1 `empty_cache()` changes the next iteration's environment

The original code called `torch.cuda.empty_cache()` after every measured pair.

It was **outside** `timed_forward`, so it is inaccurate to say its whole execution time was directly included in the printed timings. However, it changes the allocator's state before later measurements and can cause work to be repeated on subsequent allocations. That is not representative of a stable warm serving process.

**Revision:** do not clear the allocator cache inside the steady-state benchmark loop. Measure a deliberate cold-allocation experiment separately and label it differently.

### 9.2 `_` is a real Python variable, not a disposal bin

The original decode assignment was:

```python
_, decode_ms = timed_forward(...)
```

Here `_` receives the returned **model output object**, including its cache reference. Later:

```python
del outputs, past
torch.cuda.empty_cache()
```

does not necessarily release that cache, because `_` still references the decode output.

The next `for _ in range(RUNS)` iteration overwrites `_`, so this is **not proof of an ever-growing leak**. But after the final iteration, `_` still holds the last decode output. Consequently, the printed post-loop memory can include the final request's live cache.

**Revision:** use `run_idx`, `prefill_out` and `decode_out` as explicit names. Scope each request pair inside a function, and return only plain numerical measurements after intended references are released. Audit aliases before interpreting cleanup counters.

### 9.3 Warm-up covered prefill, not the full measured path

The warm-up loop invoked the model on the full prompt; it did not exercise the subsequent cached decode path.

The 128-token group's first decode was 60.07 ms versus a median of 29.56 ms. Missing decode warm-up is a plausible contributor, not a proven diagnosis of that observation.

**Revision:** warm the same prefill → select-token → cached-decode sequence that will be measured, with a fresh request cache each time. Keep cold-start measurements separately.

### 9.4 Full-sequence logits may add avoidable work

The direct forward call did not request only last-position logits. The documented Qwen2 causal-LM interface defaults to logits for every input position and supports a `logits_to_keep` option in current releases. [S15]

For generation, we consumed only the last position. Check the installed signature and actual output shape before changing this. If supported, compare `logits_to_keep=1` against the full-logits path while verifying that the last-position scores/selected tokens agree within a justified tolerance.

This is both an optimization opportunity and a **different benchmark variant**. Do not silently change it and compare against the old run as though only context length changed.

### 9.5 Current allocated memory is not peak device usage

The final `memory_allocated()` reading is one PyTorch allocator snapshot after the pair. It is not the peak during prefill, the driver's total device usage, or a direct measurement of model weights alone. [S26]

A future memory report should distinguish:

| Quantity | Intended question |
|---|---|
| Parameter/buffer accounting | What persistent model tensor storage do we expect? |
| Current allocated | What live allocator-accounted tensor allocations remain now? |
| Current reserved | What pool does PyTorch currently retain? |
| Peak allocated/reserved | How high did these counters get within a specified interval? |
| Driver-reported usage | What does the broader device-management view report? |
| Cache tensor accounting | How large is this request's K/V state specifically? |

For memory investigations, use PyTorch's memory-history/snapshot tooling when simple counters do not explain the behavior. It also has visibility limits for allocations outside the PyTorch allocator. [S19]

### 9.6 Other missing controls

The experiment did not capture a model commit, exact installed Transformers version, selected attention backend, a profile, repeated independent sessions, randomized length order, a sustained decode loop, or a no-cache correctness control. The same GPU name is not enough to guarantee identical conditions.

Five runs are enough to begin asking questions. They are not a credible basis for a production p99 claim.

### 9.7 Proposed benchmark v2 protocol — not yet executed

1. Preserve v1 code and raw observations under a distinct experiment ID.
2. Capture software versions, container digest, model/tokenizer revisions and actual parameter dtypes.
3. Confirm no unrelated GPU workload is active; record device state.
4. Define a request-pair function with explicit output/cache lifetimes.
5. Verify cached and full-context next-token computations for correctness.
6. Warm **both** prefill and cached decode for each tested shape.
7. Keep allocator behavior consistent; no `empty_cache()` inside the steady-state loop.
8. Keep the input-construction, transfer and token-selection boundaries explicitly documented.
9. Record at least 20 measured pairs per shape as an initial follow-up, preserving all samples. This is a proposed starting count, not a universal sufficiency rule.
10. Repeat blocks in a changed length order to test for drift.
11. Measure a multi-token decode sequence and peak memory in separately defined intervals.
12. Save raw CSV, metadata, source commit and limitations before making an optimization claim.

Only then vary one substantive execution setting, such as last-token-only logits. A profiler run should be identified separately from the unprofiled timing run because instrumentation can alter behavior.

## 10. First-principles memory estimates

### 10.1 Three distinct memory pools

```text
Disk:       checkpoint files, container layers, code, logs
System RAM: CPU-side objects, tokenizer output, loading buffers
GPU VRAM:   device tensors, model parameters, K/V state, GPU workspaces
```

There can be additional caches, mappings and transfers. This is a reasoning map, not a claim that every byte follows one mandatory copy sequence.

### 10.2 KV-cache size for the model configuration

For dense full-context attention using one stored K and V pair per token, a useful payload estimate is:

```text
KV bytes = 2 × layers × batch × cached_tokens × KV_heads × head_dimension × bytes_per_element
```

The factor 2 accounts for **keys and values**, not two users.

The public Qwen2.5-7B-Instruct configuration lists 28 layers, 28 query heads, 4 KV heads and hidden size 3,584. Hence head dimension is 3,584 / 28 = 128. Verify the local model's configuration and revision before treating these values as evidence about the exact downloaded snapshot. [S20]

Under BF16 cache storage and batch size 1:

```text
2 × 28 × 1 × S × 4 × 128 × 2
= 57,344 × S bytes
= 56 KiB per cached token
```

| Cached tokens | Predicted K/V tensor payload |
|---:|---:|
| 128 | 7 MiB |
| 512 | 28 MiB |
| 2,048 | 112 MiB |

One completed decode adds one processed token, so use `S + 1` for the post-decode cache. These are **derived payload estimates**, excluding allocator rounding, temporary buffers, metadata and reserved capacity.

The predicted difference between 128 and 2,048 tokens is **105 MiB**, approximately **0.1025 GiB**. That is close to the rounded 0.10-GiB change in the displayed post-loop allocation. This is a useful consistency check—especially given the surviving output/cache reference—but not independent proof that the counter measures only KV cache.

**Why query heads are not interchangeable with KV heads:** this configuration uses grouped-query attention. Using 28 query heads instead of 4 KV heads would overestimate this cache payload by 7×.

### 10.3 Logits can be large even when only one token is needed

Using vocabulary size 152,064 from the public configuration, full logits at 2,048 positions contain:

```text
1 × 2,048 × 152,064 = 311,427,072 elements
```

If those logits are BF16, their payload is approximately **0.580 GiB**; at float32 it is approximately **1.160 GiB**. Their actual dtype must be inspected rather than inferred from the model-weight dtype.

Keeping only the final position reduces the size of this *output tensor* dramatically. It does not eliminate transformer computation for the prompt, nor does it imply the same proportional reduction in total latency or peak memory.

### 10.4 Capacity is not concurrency

A tempting estimate is “free VRAM divided by KV bytes per request.” That ignores workspaces, scheduling, output-length growth, fragmentation, headroom and the latency target.

It can be one capacity bound. It cannot tell us how many simultaneous users will receive acceptable service.

## 11. Capture and publish the evidence

### 11.1 Preserve the actual VM scripts

The five scripts developed during the conversation were:

```text
01_tokenizer.py
02_load_model.py
03_inference.py
04_prefill_decode.py
05_context_benchmark.py
```

Copy the **actual files from the VM** into the repository. These Markdown files do not pretend to be an exact export of those source files or their execution history. Preserve the original benchmark and add a revised version as a separate change.

Suggested repository layout:

```text
ai-systems-performance-lab/
  README.md
  labs/
    day01-baseline-execution-map/
      DAY01_LAB_NOTES.md
      DAY01_CURIOSITY_QUESTIONS.md
      01_tokenizer.py
      02_load_model.py
      03_inference.py
      04_prefill_decode.py
      05_context_benchmark.py
      results/
        context_v1_transcribed.csv
        environment.json
        host_environment.txt
        requirements-observed.txt
  articles/
```

The relative companion link assumes the two Markdown files remain in the same folder. The other files in this tree are suggested destinations—not a claim that they were generated with this document.

### 11.2 Capture missing environment details now

These are **proposed evidence-capture commands**, not historical output. Run the first block on the Ubuntu host:

```bash
mkdir -p "$HOME/ai-systems-lab/results/day01"
{
  date -u +%FT%TZ
  cat /etc/os-release
  uname -r
  lscpu
  free -h
  df -h /
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
  docker --version
  nvidia-container-cli --version
  sudo docker image inspect pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime \
    --format '{{json .RepoDigests}}'
} > "$HOME/ai-systems-lab/results/day01/host_environment.txt" 2>&1
```

Capture a targeted package report inside `gpu-lab`:

```bash
mkdir -p /workspace/results/day01
python - <<'PY'
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import torch

packages = {}
for name in ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub"):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None

cuda_available = torch.cuda.is_available()
report = {
    "captured_at_stage": "post_day01_followup_not_original_benchmark_manifest",
    "python": platform.python_version(),
    "packages": packages,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "original_model_revision": "NOT_RECORDED",
    "original_attention_backend": "NOT_RECORDED",
}
path = Path("/workspace/results/day01/environment.json")
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"Saved {path}")
PY

python -m pip freeze --all > /workspace/results/day01/requirements-observed.txt
```

Review the freeze output before publishing: direct-install URLs can contain local paths or credentials. An observed package inventory is useful, but is not automatically a cross-platform dependency lockfile.

When collecting future benchmark output, use `tee` to retain it. Do not overwrite the original results with a new execution under the same filename. A rerun is a new experiment with a new timestamp/identifier.

### 11.3 Public-repository safety check

Before staging files, exclude private keys, access tokens, account/project IDs, unnecessary public IPs, model weights, full caches and unredacted infrastructure screenshots. Review selected files and `git diff --cached` before committing.

A starting ignore policy:

```gitignore
*.pem
*.key
.env
.env.*
!.env.example
**/.ssh/
**/__pycache__/
*.py[cod]
**/.cache/
**/hf-cache/
models/
*.safetensors
*.pt
*.pth
```

Ignore patterns are not a secret scanner. They also do not remove files already tracked or exposed in repository history. If a credential is accidentally committed, revoke/rotate it and follow the repository host's remediation guidance. [S21]

### 11.4 An accurate public summary

> I built a single-GPU inference lab on a spot VM and verified the path from host driver and container access to PyTorch tensors, tokenization, model prefill and cached decoding. In an exploratory batch-1 context sweep, prefill forward latency rose from 36.30 ms at 128 tokens to 304.57 ms at 2,048 tokens; one-token cached decode forward latency stayed close to 30 ms. These are synchronized model-forward measurements, not end-to-end serving latency. A code review identified warm-up, allocator-state and reference-lifetime issues that will be corrected in the next benchmark revision.

That summary is stronger than claiming an unmeasured speedup or an enterprise platform. The correction itself is an engineering result.

## 12. What is complete and what comes next

### 12.1 Day-1 completion record

- [x] Cloud connectivity and successful SSH access demonstrated.
- [x] Host driver and container GPU visibility demonstrated.
- [x] Python/PyTorch CUDA execution demonstrated.
- [x] Tensor memory calculation and allocation exercise completed; exact large-allocation output not archived here.
- [x] Tokenizer output inspected on CPU.
- [x] BF16 model execution and one cached decode step demonstrated.
- [x] Exploratory context-length measurements obtained and retained in this note.
- [x] Benchmark limitations identified instead of hidden.
- [ ] Exact source, package versions, image digest and model revision packaged for independent reproduction.
- [ ] Corrected benchmark run with a correctness check and complete warm-up.
- [ ] Independent reproduction on a fresh environment.
- [ ] Controlled multi-user serving experiment.
- [ ] Profiler-supported bottleneck diagnosis and validated optimization.

**Discussed, not implemented here:** vLLM, continuous batching, a production API, Kubernetes, multi-GPU communication, a golden image, CUDA C++ kernels, quantization comparisons and load-tested service-level objectives.

### 12.2 Three next actions, in order

**First: preserve and review.** Commit the original source, transcribed results and these notes after the security review. Capture missing metadata without pretending it was recorded earlier.

**Second: repair measurement.** Implement benchmark v2 and a cache-correctness control before assigning a cause to the nearly flat decode curve.

**Third: introduce two users.** Begin with one model instance and two request records. Compare sequential scheduling with static batching, documenting arrival time, queue wait, completion time and per-user isolation. Introduce a serving engine only after the elementary comparison makes sense.

For a decoder-only model, static batched generation usually requires attention to **left padding**, attention masks and model-specific positions. A naïve right-padded batch plus `logits[:, -1, :]` can select scores at a pad position for shorter prompts. Correctness comes before a speed comparison. [S16], [S22]

Do not add Docker layers or buy another GPU as an answer to every performance question. Ask which resource or dependency is actually limiting the workload.

### 12.3 Reflection prompts

Write personal answers before opening the companion workbook:

1. Which error changed my understanding of a system boundary?
2. Which fact did I initially assume but later measure?
3. Which graph could I have misinterpreted without reviewing the code?
4. What remains a hypothesis rather than a demonstrated cause?
5. What is the smallest experiment that could prove my favorite explanation wrong?

> **Day-1 outcome:** a verified single-GPU execution path, an honest record of failures and measurements, and a sharper set of questions—not a claim that the system is already optimized.

## 13. Sources

Primary documentation was consulted while preparing these notes. Documentation may describe newer releases than the lab's installed environment; version-sensitive APIs must be checked locally. Lab measurements are sourced from the learner's shared terminal output, not from these references.

- **S01 — NVIDIA RTX A6000 specifications:** https://www.nvidia.com/en-us/products/workstations/rtx-a6000/
- **S02 — NVIDIA System Management Interface:** https://docs.nvidia.com/deploy/nvidia-smi/index.html
- **S03 — NVIDIA Container Toolkit architecture:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html
- **S04 — NVIDIA CUDA compiler driver (`nvcc`):** https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- **S05 — Docker architecture and concepts:** https://docs.docker.com/get-started/docker-overview/
- **S06 — Kubernetes container runtimes:** https://kubernetes.io/docs/setup/production-environment/container-runtimes/
- **S07 — OpenSSH `ssh-keygen`:** https://man.openbsd.org/ssh-keygen
- **S08 — PowerShell character encoding:** https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding
- **S09 — Ubuntu automatic updates:** https://ubuntu.com/server/docs/how-to/software/automatic-updates/
- **S10 — NVIDIA Container Toolkit installation:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- **S11 — Docker `exec`:** https://docs.docker.com/reference/cli/docker/container/exec/
- **S12 — Transformers model loading:** https://huggingface.co/docs/transformers/en/main_classes/model
- **S13 — Docker bind mounts:** https://docs.docker.com/engine/storage/bind-mounts/
- **S14 — PyTorch 2.7 CUDA semantics:** https://docs.pytorch.org/docs/2.7/notes/cuda.html
- **S15 — Qwen2 model interface:** https://huggingface.co/docs/transformers/en/model_doc/qwen2
- **S16 — Transformers chat templates:** https://huggingface.co/docs/transformers/en/chat_templating
- **S17 — Transformers cache explanation:** https://huggingface.co/docs/transformers/en/cache_explanation
- **S18 — vLLM serving benchmark metrics:** https://docs.vllm.ai/en/latest/cli/bench/serve/
- **S19 — PyTorch CUDA memory investigation:** https://docs.pytorch.org/docs/2.7/torch_cuda_memory.html
- **S20 — Qwen2.5-7B-Instruct configuration:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
- **S21 — GitHub guidance for sensitive-data exposure:** https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
- **S22 — Transformers LLM generation guidance:** https://huggingface.co/docs/transformers/en/llm_tutorial

- **S23 — Podman architecture:** https://docs.podman.io/en/latest/
- **S24 — NVIDIA CDI support:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
- **S25 — PyTorch 2.7 `empty_cache`:** https://docs.pytorch.org/docs/2.7/generated/torch.cuda.empty_cache.html
- **S26 — PyTorch 2.7 `memory_allocated`:** https://docs.pytorch.org/docs/2.7/generated/torch.cuda.memory_allocated.html
- **S27 — PyTorch 2.7 embedding lookup:** https://docs.pytorch.org/docs/2.7/generated/torch.nn.Embedding.html
- **S28 — PyTorch 2.7 CUDA events:** https://docs.pytorch.org/docs/2.7/generated/torch.cuda.Event.html

<!-- Reference definitions for clickable inline source labels. -->

[S01]: https://www.nvidia.com/en-us/products/workstations/rtx-a6000/
[S02]: https://docs.nvidia.com/deploy/nvidia-smi/index.html
[S03]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html
[S04]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
[S05]: https://docs.docker.com/get-started/docker-overview/
[S06]: https://kubernetes.io/docs/setup/production-environment/container-runtimes/
[S07]: https://man.openbsd.org/ssh-keygen
[S08]: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding
[S09]: https://ubuntu.com/server/docs/how-to/software/automatic-updates/
[S10]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[S11]: https://docs.docker.com/reference/cli/docker/container/exec/
[S12]: https://huggingface.co/docs/transformers/en/main_classes/model
[S13]: https://docs.docker.com/engine/storage/bind-mounts/
[S14]: https://docs.pytorch.org/docs/2.7/notes/cuda.html
[S15]: https://huggingface.co/docs/transformers/en/model_doc/qwen2
[S16]: https://huggingface.co/docs/transformers/en/chat_templating
[S17]: https://huggingface.co/docs/transformers/en/cache_explanation
[S18]: https://docs.vllm.ai/en/latest/cli/bench/serve/
[S19]: https://docs.pytorch.org/docs/2.7/torch_cuda_memory.html
[S20]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
[S21]: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
[S22]: https://huggingface.co/docs/transformers/en/llm_tutorial
[S23]: https://docs.podman.io/en/latest/
[S24]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
[S25]: https://docs.pytorch.org/docs/2.7/generated/torch.cuda.empty_cache.html
[S26]: https://docs.pytorch.org/docs/2.7/generated/torch.cuda.memory_allocated.html
[S27]: https://docs.pytorch.org/docs/2.7/generated/torch.nn.Embedding.html
[S28]: https://docs.pytorch.org/docs/2.7/generated/torch.cuda.Event.html
