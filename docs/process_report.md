# ARM AI Optimization Challenge — Process Report

## 1. Project Overview

**Challenge:** ARM AI Optimization Track — optimize an LLM for on-device inference on constrained ARM hardware.

**Goal:** Deploy Qwen2-0.5B on a Raspberry Pi 5 to perform real-time triage of IoT sensor time-series data (temperature, humidity, people count) with p99 latency < 2000 ms.

**Pipeline:**
```
Sensor CSV → 30s sliding window → threshold rules (<1 ms) → LLM triage (Qwen2-0.5B) → JSON alert
```

**Hardware targets:**
- Raspberry Pi 5, 8 GB RAM, 4× Cortex-A76 @ 2.4 GHz
- Hailo-8 AI NPU (PCIe accelerator, present but not usable for LLM)
- Dev machine: Ubuntu 24.04, AMD Ryzen, RTX 4090 (used for model conversion/quantization only)

---

## 2. Decision Process

### 2.1 Inference Framework: llama.cpp vs ONNX Runtime

**Decision: llama.cpp + GGUF**

| Criteria               | llama.cpp + GGUF              | ONNX Runtime + ARM NN         |
|------------------------|-------------------------------|-------------------------------|
| LLM optimizations      | Purpose-built, NEON intrinsics| General DNN, no LLM focus    |
| Binary size            | < 10 MB (minimal build)       | > 100 MB with ARM NN deps     |
| Quantization           | GGUF Q4_K_M (state-of-art)    | ONNX INT8 (limited support)  |
| Aarch64/NEON           | First-class, -DGGML_ARM64=ON  | Partial via ARM NN            |
| Qwen2 support          | Native (llama.cpp supports)   | Requires custom export        |
| Community              | Large, active (ggerganov)     | Smaller for ARM inference     |

**Winner:** llama.cpp — purpose-built for LLM inference, smallest binary, best ARM support.

### 2.2 Base Model: Qwen2-0.5B

Selected for:
- **Size:** 494M params → 380 MB at Q4_K_M (fits in 8 GB RAM with overhead)
- **Architecture:** Standard transformer decoder (well-supported by llama.cpp)
- **Performance:** Balances accuracy vs inference speed on Cortex-A76
- **License:** Apache 2.0 (permissive for competition)

Alternatives considered:
- Phi-3-mini (3.8B): Too large for on-device real-time
- SmolLM2 (135M): Too small for reliable triage
- Llama-3.2-1B: Viable but larger (1B vs 0.5B) and similar accuracy

### 2.3 Quantization: Q4_K_M

| Format | Size | Speed | Quality |
|--------|------|-------|---------|
| FP16   | 949 MB | Baseline | Baseline |
| Q8_0   | 507 MB | ~1.2× | Near-lossless |
| **Q4_K_M** | **380 MB** | **~1.5×** | **Good (K-quants)** |

Q4_K_M selected as the best size/accuracy tradeoff for a 494M-parameter model on 8 GB RAM.

### 2.4 Prompt Engineering: Few-Shot Completion

Qwen2-0.5B at Q4_K_M **cannot follow JSON instructions** reliably. The few-shot completion approach provides 6 worked examples and asks the model to complete the pattern, resulting in reliable JSON output without parse errors.

---

## 3. Implementation

### 3.1 Project Structure

```
arm-optimize-iot/
├── src/
│   ├── alerting/
│   │   ├── alert_actions.py      # Alert dispatch (log, GPIO, notify)
│   │   ├── alert_engine.py       # Hybrid threshold + LLM pipeline
│   │   ├── latency_monitor.py    # p50/p95/p99 budget enforcement
│   │   ├── llm_triage.py         # Qwen2-0.5B few-shot prompting
│   │   └── threshold_rules.py    # 6 fast rule checks (<1 ms)
│   ├── data_processing/
│   │   ├── sensor_generator.py   # Synthetic CSV with 4 anomaly types
│   │   └── sensor_reader.py      # Sliding window, CSV/JSON/SQLite
│   └── model_optimization/
│       ├── benchmark.py          # Inference speed benchmark
│       ├── download_model.py     # HF hub download
│       ├── export_to_gguf.py     # HF → GGUF conversion script
│       ├── quantize_model.py     # Q4_K_M quantization
│       └── optimize_model.py     # Full pipeline orchestrator
├── tests/
│   ├── test_latency_monitor.py   # 6 tests
│   ├── test_sensor_generator.py  # 12 tests
│   ├── test_sensor_reader.py     # 9 tests
│   └── test_threshold_rules.py   # 8 tests
├── models/                       # GGUF model files
├── data/                         # Generated sensor CSV
├── docs/
│   ├── benchmark_results.md      # Full benchmark report
│   └── rpi5_deployment.md        # Deployment guide
├── setup.sh                      # One-command environment setup
└── benchmark.sh                  # Automated benchmark suite
```

### 3.2 Key Files

#### `src/alerting/threshold_rules.py`
- 6 rule types: temp_high, temp_low, humidity_high, humidity_low, people_high, temp_rise
- Configurable thresholds via `ThresholdConfig` dataclass
- Returns list of `TriggeredRule` dataclasses

#### `src/alerting/llm_triage.py`
- `LLMTriageEngine` class wrapping `llama-cpp-python`
- Few-shot prompt with 6 sensor→JSON examples
- Robust JSON parsing with bracket-finding fallback
- Configurable `n_threads`, `n_ctx`, `max_tokens`

#### `src/alerting/alert_engine.py`
- `AlertEngine` orchestrator: threshold → LLM → alert
- Lazy LLM loading (only on first threshold trigger)
- `AlertResult` dataclass with timing breakdown
- `main()` entry point: `python -m src.alerting.alert_engine --data path.csv`

#### `src/alerting/latency_monitor.py`
- Tracks p50, p95, p99, mean, min, max
- Configurable budget (default: 2000 ms)
- Tracks exceeded count

#### `src/data_processing/sensor_generator.py`
- Generates realistic diurnal temperature/humidity patterns
- Injects 4 anomaly types: temp spikes, humidity drops, people surges, alarm triggers
- Reproducible via `--seed`

#### `src/data_processing/sensor_reader.py`
- `SensorWindow` dataclass with summary statistics
- `SensorReader` for CSV/JSON/SQLite3 via `--format`
- Sliding window via `stream(window_s, step_s)`

---

## 4. Model Pipeline

### 4.1 Steps

1. **Download** Qwen2-0.5B-Instruct from Hugging Face (953 MB safetensors)
2. **Convert** to GGUF FP16 using llama.cpp's `convert_hf_to_gguf.py` (949 MB)
3. **Quantize** to Q4_K_M using `llama-quantize` (380 MB, 40% of original)
4. **Benchmark** with `llama-bench` and `llama-cpp-python`

All steps performed on dev machine (RTX 4090, 32 GB RAM). Quantization completed in ~2 minutes.

---

## 5. Deployment

### 5.1 Target: Raspberry Pi 5

**SSH access:** Tailscale network (`icp@smartshelf.taila277ca.ts.net`, password auth)

**Existing environment:** Production application `agentic_monitoring.py` running in `~/g_traffic/` (uses Hailo-8 for person detection). Our project deployed to a separate directory `~/arm-optimization-iot/` to avoid interference.

**Setup steps:**
```bash
# On RPi5 (via SSH)
python3 -m venv ~/arm-optimization-iot/venv
source venv/bin/activate
pip install numpy pandas pytest pyyaml psutil
CMAKE_ARGS='-DGGML_ARM64=ON -DGGML_NATIVE=ON' pip install llama-cpp-python

# Transfer model (380 MB over Tailscale, ~2.4 MB/s, ~2:40)
scp qwen2-0.5b-q4_k_m.gguf icp@...:~/arm-optimization-iot/models/

# Build llama.cpp natively for optimal performance
git clone https://github.com/ggml-ai/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_ARM64=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
make -j4 llama-bench llama-cli
```

### 5.2 Testing

```bash
# Unit tests (34/34 pass)
cd ~/arm-optimization-iot && python3 -m pytest tests/ -v

# Model benchmark
python3 -m src.model_optimization.benchmark

# Native benchmark
LD_LIBRARY_PATH=llama.cpp-src/build-native/bin \
  llama.cpp-src/build-native/bin/llama-bench \
  -m models/qwen2-0.5b-q4_k_m.gguf -p 64 -n 256 -t 4

# End-to-end alert pipeline
python3 -m src.data_processing.sensor_generator --hours 1 --output data/test.csv
python3 -m src.alerting.alert_engine --data data/test.csv
```

---

## 6. Results

### 6.1 Test Suite

**34/34 tests passing on RPi5** (completed in 3.94s):
- test_latency_monitor: 6/6
- test_sensor_generator: 12/12
- test_sensor_reader: 9/9
- test_threshold_rules: 8/8

### 6.2 Inference Performance

| Metric                    | 1 Thread     | 4 Threads    |
|---------------------------|-------------|--------------|
| Prompt processing (64 tok)| 24.35 tok/s | 91.77 tok/s  |
| Text generation (256 tok) | 16.67 tok/s | 28.38 tok/s  |
| TTFT (est.)               | ~2.6 s      | ~0.7 s       |
| Time to generate 50 tok   | ~3.0 s      | ~1.8 s       |

### 6.3 Alert Pipeline

| Metric            | Threshold-only | LLM triage     | Combined       |
|-------------------|---------------|----------------|----------------|
| p50 latency       | < 0.1 ms      | 31,991 ms      | 31,991 ms      |
| p99 latency       | < 0.1 ms      | 42,043 ms      | 42,043 ms      |
| Budget (2,000 ms) | ✓             | ✗              | ✗              |

### 6.4 Hailo AI NPU

**Present but not usable for LLM.** The Hailo-8 is a CNN accelerator designed for computer vision:
- `/dev/hailo0` device node present
- `hailo_pci` kernel driver loaded
- `hailort 4.21.0` Python package available
- Used for YOLO-based person detection in the existing app
- **No llama.cpp backend exists** — cannot accelerate transformer/LLM inference

---

## 7. Key Findings

### 7.1 What Worked

- ✅ **Hybrid threshold + LLM architecture** — threshold rules handle 80%+ of windows in <0.1 ms
- ✅ **Native llama.cpp build** — 7× faster than llama-cpp-python (28 vs 4 tok/s)
- ✅ **Q4_K_M quantization** — 380 MB model fits comfortably in 8 GB RAM
- ✅ **Few-shot completion prompting** — 100% valid JSON output from Qwen2-0.5B
- ✅ **HAILO NPU analysis** — documented that it doesn't accelerate LLM

### 7.2 What Didn't

- ❌ **p99 < 2000 ms budget** — LLM triage takes 5-42s per call, well above budget
- ❌ **llama-cpp-python threading** — Python GIL makes 4 threads slower than 1
- ❌ **Pip-installed llama-cpp-python** — generic ARM64 binary lacks proper NEON optimization
- ❌ **Dev machine GPU** — no NVIDIA driver loaded, couldn't compare GPU vs CPU

### 7.3 Recommendations for Budget Compliance

1. **Subprocess llama-cli** — Replace `llama-cpp-python` with native `llama-cli` subprocess calls (sub-2s per call achievable)
2. **Reduce generation tokens** — `max_tokens=20` instead of 50 halves latency
3. **Prompt caching** — Skip LLM for identical window patterns
4. **Selective LLM triage** — Only run LLM on high-severity threshold triggers
5. **Smaller model** — Try SmolLM2-360M or similar at Q4_K_M

---

## 8. Files on RPi5

```
/home/icp/arm-optimization-iot/
├── models/qwen2-0.5b-q4_k_m.gguf    # 380 MB production model
├── llama.cpp-src/                    # Native llama.cpp build (28 MB)
│   └── build-native/bin/
│       ├── llama-bench               # ARM aarch64 native binary
│       ├── llama-cli                 # ARM aarch64 native binary
│       └── libggml-cpu.so            # 908 KB NEON-optimized library
├── src/                              # Project source
├── tests/                            # 34 unit tests
├── venv/                             # Python venv with llama-cpp-python
└── data/                             # Generated sensor CSV
```

## 9. Files on Dev Machine

```
/home/stanl/projects/arm-optimize-iot/
├── models/
│   ├── qwen2-0.5b-hf/               # Original HF model (953 MB)
│   ├── qwen2-0.5b-fp16.gguf         # FP16 GGUF (949 MB)
│   ├── qwen2-0.5b-q8_0.gguf         # Q8_0 GGUF (507 MB)
│   └── qwen2-0.5b-q4_k_m.gguf       # Q4_K_M GGUF (380 MB) ← production
├── src/                              # Project source
├── tests/                            # 34 unit tests
├── venv/                             # Python 3.12 venv
├── docs/
│   ├── benchmark_results.md          # Full results (this document)
│   └── rpi5_deployment.md            # Deployment guide
├── setup.sh                          # One-command setup
├── benchmark.sh                      # Benchmark suite
└── README.md                         # Project overview
```
