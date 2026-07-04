# ARM AI Optimization Challenge — Submission

**Track:** Track 1 — Physical AI (On-Device Edge AI)

**Project:** Hybrid Threshold + LLM Triage Pipeline on Raspberry Pi 5

**Repository:** [https://github.com/stanleyoz/arm-optimize-iot](https://github.com/stanleyoz/arm-optimize-iot)

**License:** Apache 2.0

---

## Project Overview

### What is this?

A real-time IoT sensor anomaly detection system that runs a **500M-parameter language model (Qwen2-0.5B)** on a **Raspberry Pi 5** to triage time-series data from temperature, humidity, and people-count sensors — all within a **2-second latency budget**.

The core insight is a **hybrid architecture**: fast rule-based threshold checks (<0.1 ms) filter out 82% of normal windows, while a quantized LLM (380 MB Q4_K_M) analyzes the remaining ambiguous cases with contextual reasoning. This avoids the cost of running the LLM on every window while retaining intelligence for complex edge cases.

### Why is it interesting?

1. **LLM on $80 ARM hardware** — Demonstrates that a 0.5B-parameter transformer can deliver useful inference on a Cortex-A76 CPU at 4W TDP, without GPU or NPU acceleration. This challenges the assumption that edge AI requires specialized hardware.

2. **Systematic optimization, not just model swap** — The 5.8s → 1.8s p99 latency improvement came from a chain of independent optimizations (quantization 2.5×, thread tuning 3.8×, token budget 5.3×, cold-start elimination 3.2×, hybrid pipeline 5.5× system-level). Each is measurable and reproducible.

3. **Prompt engineering replaces fine-tuning** — A few-shot completion strategy achieved 100% valid JSON output from a 4-bit quantized model without any LoRA or RLHF. The model was rewired purely through prompt structure.

4. **Honest NPU analysis** — We tested and documented that the on-board Hailo-8 AI accelerator cannot accelerate LLM inference (CNN-only architecture). This saves other developers from pursuing a dead end.

### Why it should win

| Judging Criterion | How we hit it |
|---|---|
| **Technological Implementation (40 pts)** | 1,000+ lines of production-quality Python with 34 unit tests; systematic optimization with p50/p95/p99 measurements at every step; clean modular architecture |
| **User/Developer Experience (15 pts)** | Single `./setup.sh` builds everything; `./benchmark.sh` reproduces all results; `SensorReader` accepts CSV/JSON/SQLite3; full deployment guide for RPi5 |
| **Potential Impact (20 pts)** | Reusable template for any sensor-to-LLM edge pipeline; proves LLM viability on commodity ARM hardware; saves NPU exploration dead-end |
| **WOW Factor (25 pts)** | **500M-parameter LLM doing real-time sensor triage on a $80 Raspberry Pi 5 — 41/41 alerts under 2 seconds, p99=1,841ms, 0 budget exceeded** |

---

## Functionality / Output

### What it does

```
[Sensor CSV stream] ──> [30s sliding window] ──> [Threshold rules: <0.1ms]
                                                         │
                                                   82% pass ──> No alert
                                                         │
                                                   18% trigger
                                                         │
                                              ┌──────────┴──────────┐
                                              │ severity = "low"    │
                                              │ (people count only) │
                                              │ ──> Alert directly  │
                                              │     (<0.1ms)        │
                                              └─────────────────────┘
                                                         │
                                              medium/high severity
                                                         │
                                              [LLM Triage: ~1.8s]
                                                         │
                                              [JSON decision output]
                                                         │
                                              [AlertLogger: log/GPIO]
```

### Final output

Each processed sensor window produces an `AlertResult` dataclass:

```
AlertResult(
    alert=True,
    reason="Temperature 42C exceeds threshold and room is occupied",
    severity="high",
    threshold_time_ms=0.08,
    llm_time_ms=1812.3,
    total_time_ms=1812.4,
    triggered_rules=[TriggeredRule(name="temp_high", ...)],
    llm_raw={"alert": True, "reason": "...", "severity": "high"},
    latency_budget_exceeded=False,
)
```

The LLM always produces **valid JSON** with three fields: `alert` (bool), `reason` (string), `severity` ("low"/"medium"/"high"). The system degrades gracefully via brace-depth parsing and regex fallback if output is truncated.

### Measured performance (Raspberry Pi 5, 4-hour synthetic data, 55 triggered windows)

| Metric | Threshold | LLM Triage | Total per Alert |
|---|---|---|---|
| p50 | <0.1 ms | 1,780 ms | 1,780 ms |
| p95 | <0.1 ms | 1,826 ms | 1,826 ms |
| **p99** | **<0.1 ms** | **1,841 ms** | **1,841 ms** |
| **Max** | **<0.1 ms** | **1,841 ms** | **1,841 ms** |
| Budget exceeded | 0/55 | 0/55 | **0/55** |

### All optimization gains at a glance

| Optimization | Before | After | Improvement |
|---|---|---|---|
| FP16 → Q4_K_M quantization | 949 MB, — tok/s | 380 MB, 16.7 tok/s | 2.5× size, ~1.5× speed |
| GIL-aware `n_threads=1` | 4.0 tok/s (4 threads) | 15.1 tok/s (1 thread) | 3.8× generation speed |
| `max_tokens=128` → 25 | ~8.5s/call | ~1.6s/call | 5.3× per-call reduction |
| Lazy load → eager + warmup | 5,833 ms p99 | 1,841 ms p99 | 3.2× cold-start |
| Hybrid threshold gate | — | ~324 ms effective avg | 5.5× system-level |
| Generic pip → native NEON build | 4.0 tok/s | 28.4 tok/s (4T native) | 7× prompt processing |

---

## Setup Instructions

### Prerequisites

- **Hardware:** Raspberry Pi 5 (4GB+ RAM), or any ARM64 Linux system with aarch64 CPU
- **OS:** Raspberry Pi OS (Bookworm) or Ubuntu 24.04 for ARM64
- **Storage:** 2 GB free for code + model
- **Python:** 3.10+

### Step 1 — Clone and setup

```bash
git clone https://github.com/stanleyoz/arm-optimize-iot.git
cd arm-optimize-iot
./setup.sh
```

`setup.sh` will:
1. Create a Python virtual environment
2. Install Python dependencies (llama-cpp-python, numpy, pandas, etc.)
3. Clone and build llama.cpp from source with NEON optimizations (`-DLLAMA_NATIVE=ON`)
4. Symlink llama.cpp tools into `bin/`

### Step 2 — Download and quantize the model

```bash
# Activate environment
source venv/bin/activate

# Option A: Download pre-quantized model from Hugging Face
python -m src.model_optimization.download_model

# Option B: Full pipeline (download HF → convert → quantize)
python -m src.model_optimization.optimize_model
```

This produces `models/qwen2-0.5b-q4_k_m.gguf` (380 MB).

### Step 3 — Generate sensor data

```bash
python -m src.data_processing.sensor_generator --hours 24 --output data/sensor_data.csv
```

Generates synthetic CSV with temperature, humidity, people_count, and 4 anomaly types (temp_spike, humidity_spike, people_surge, sensor_fault).

### Step 4 — Run the alert pipeline

```bash
# Quick demo (1 hour of data)
python -m src.alerting.alert_engine
```

Or with custom data:

```bash
python -c "
from src.data_processing.sensor_reader import SensorReader
from src.alerting.alert_engine import AlertEngine

reader = SensorReader('data/sensor_data.csv')
engine = AlertEngine()
for window in reader.stream(window_s=30, step_s=30):
    result = engine.process(window)
    if result.alert:
        print(f'{result.severity.upper()}: {result.reason}  [{result.total_time_ms:.0f}ms]')
engine.close()
"
```

### Step 5 — Run benchmarks

```bash
./benchmark.sh
```

This runs:
- Model benchmark (tokens/s, TTFT, memory via `llama-bench`)
- All 34 unit tests
- End-to-end alert pipeline on generated data

### Step 6 — Validate latency budget

```bash
python -c "
import csv, time
from src.data_processing.sensor_generator import generate_sensor_data
from src.data_processing.sensor_reader import SensorReader
from src.alerting.alert_engine import AlertEngine

rows = generate_sensor_data(hours=4, seed=99)
with open('/tmp/validate.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

engine = AlertEngine()
reader = SensorReader('/tmp/validate.csv')
times = []
for w in reader.stream(window_s=30, step_s=30):
    r = engine.process(w)
    if r.llm_time_ms > 0:
        times.append(r.llm_time_ms)

times.sort()
n = len(times)
print(f'LLM calls: {n}')
print(f'p50: {times[n//2]:.1f}ms')
print(f'p95: {times[int(n*0.95)]:.1f}ms')
print(f'p99: {times[int(n*0.99)]:.1f}ms')
print(f'max: {max(times):.1f}ms')
print(f'Under 2000ms: {sum(1 for t in times if t < 2000)}/{n}')
engine.close()
"
```

Expected output:
```
LLM calls: ~41
p50: ~1780ms
p95: ~1825ms
p99: ~1840ms
max: ~1840ms
Under 2000ms: 41/41
```

---

## Repository Structure

```
arm-optimize-iot/
├── LICENSE                     # Apache 2.0
├── README.md                   # This file
├── setup.sh                    # One-command environment build
├── benchmark.sh                # Automated benchmark suite
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── submission/
│   └── SUBMISSION.md           # This submission write-up
│
├── src/
│   ├── alerting/
│   │   ├── alert_actions.py    # Alert dispatch (log, GPIO, notify)
│   │   ├── alert_engine.py     # Hybrid orchestrator (threshold → LLM)
│   │   ├── latency_monitor.py  # p50/p95/p99 budget enforcement
│   │   ├── llm_triage.py       # Qwen2-0.5B few-shot prompting
│   │   └── threshold_rules.py  # 6 fast rule checks
│   ├── data_processing/
│   │   ├── sensor_generator.py # Synthetic CSV with 4 anomaly types
│   │   └── sensor_reader.py    # Sliding window, CSV/JSON/SQLite
│   └── model_optimization/
│       ├── benchmark.py        # Inference speed benchmark
│       ├── download_model.py   # HF hub download
│       ├── export_to_gguf.py   # HF → GGUF conversion
│       ├── quantize_model.py   # Q4_K_M quantization
│       └── optimize_model.py   # Full pipeline orchestrator
│
├── tests/
│   ├── test_latency_monitor.py # 6 tests
│   ├── test_sensor_generator.py# 12 tests
│   ├── test_sensor_reader.py   # 9 tests
│   └── test_threshold_rules.py # 8 tests
│
├── docs/
│   ├── benchmark_results.md    # Full benchmark report
│   ├── process_report.md       # Decision process documentation
│   └── rpi5_deployment.md      # Deployment guide for RPi5
│
├── models/                     # GGUF model files (gitignored)
├── data/                       # Generated sensor CSV (gitignored)
└── llama.cpp/                  # llama.cpp source (gitignored build/)
```

---

## Optional Video

*[Link to YouTube/Vimeo demo video — <3 minutes showing the pipeline running on Raspberry Pi 5 with real-time alert output]*

---

## Third-Party Integrations

| Component | License | Usage |
|---|---|---|
| Qwen2-0.5B (Qwen) | Apache 2.0 | Base model, downloaded from Hugging Face |
| llama.cpp (ggerganov) | MIT | Inference engine, built from source |
| llama-cpp-python (abetlen) | MIT | Python bindings for llama.cpp |
| PyTorch | BSD-3 | Model conversion (dev machine only) |
| Hugging Face Transformers | Apache 2.0 | Model download (dev machine only) |
| Hugging Face Hub | Apache 2.0 | Model download |
| NumPy, pandas, scikit-learn | BSD-3 | Data processing |
