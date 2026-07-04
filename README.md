# ARM AI Optimization Challenge — Hybrid Edge LLM for Sensor Triage

**Track:** Track 1 — Physical AI (On-Device Edge AI)

**Base Model:** Qwen2-0.5B → GGUF Q4_K_M (380 MB)  
**Inference Engine:** llama.cpp built from source with NEON for Cortex-A76  
**Target Device:** Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz, 8 GB RAM)  
**Pipeline:** Threshold rules (<0.1ms) → LLM triage (Qwen2-0.5B, ~1.8s) → JSON alert

**Key Result:** p99 LLM triage latency **1,841 ms** — 41/41 alerts under the 2-second budget, 0 budget exceeded.

---

## Architecture

```
[Sensor CSV stream] → [30s sliding window] → [Threshold rules: <0.1ms]
                                                    │
                                              82% pass → No alert
                                                    │
                                              18% trigger
                                                    │
                                         ┌──────────┴──────────┐
                                         │ severity = "low"    │
                                         │ (people count only) │
                                         │ → Alert directly    │
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

## Quick Start

```bash
git clone <repo> && cd arm-optimize-iot
./setup.sh                                    # Build env + llama.cpp
source venv/bin/activate
python -m src.model_optimization.download_model  # Get Qwen2-0.5B Q4_K_M
python -m src.data_processing.sensor_generator   # Generate test data
python -m src.alerting.alert_engine              # Run pipeline
./benchmark.sh                                   # Full benchmark suite
```

## Optimization Summary

| Optimization | Before | After | Gain |
|---|---|---|---|
| FP16 → Q4_K_M quantization | 949 MB / 949 MB RAM | 380 MB / 297 MB RAM | 2.5× memory reduction |
| GIL-aware `n_threads=1` | 4.0 tok/s (4 threads) | 15.1 tok/s (1 thread) | 3.8× generation speed |
| `max_tokens=128` → 25 | ~8.5s per call | ~1.6s per call | 5.3× per-call reduction |
| Lazy load → eager + warmup | 5,833 ms p99 | 1,841 ms p99 | 3.2× cold-start elimination |
| Hybrid pipeline gate | — | ~324 ms effective avg | 5.5× system-level |
| Generic pip → native NEON build | 4.0 tok/s | 28.4 tok/s (4T native) | 7× prompt processing |

## Benchmark Results (Raspberry Pi 5)

| Metric | Threshold Check | LLM Triage | Total per Alert |
|---|---|---|---|
| p50 | <0.1 ms | 1,780 ms | 1,780 ms |
| p95 | <0.1 ms | 1,826 ms | 1,826 ms |
| **p99** | **<0.1 ms** | **1,841 ms** | **1,841 ms** |
| Max | <0.1 ms | 1,841 ms | 1,841 ms |
| Budget exceeded | 0/55 | 0/55 | **0/55** |

## Project Structure

```
src/
  alerting/          # Core pipeline: threshold_rules, llm_triage, alert_engine
  data_processing/   # sensor_generator, sensor_reader (CSV/JSON/SQLite)
  model_optimization/# download, export, quantize, benchmark
tests/               # 34 unit tests, all passing
docs/                # benchmark_results, process_report, rpi5_deployment
submission/          # Devpost submission write-up
setup.sh             # One-command environment + llama.cpp build
benchmark.sh         # Automated benchmark suite
LICENSE              # Apache 2.0
```

## Key Technical Decisions

1. **Qwen2-0.5B** at Q4_K_M — smallest size with reliable JSON output; 380 MB fits Pi 5 RAM
2. **Few-shot completion** instead of instruction prompting — 100% JSON compliance at 4-bit (instruction mode fails at this quantization level)
3. **`n_threads=1`** in llama-cpp-python — Python GIL makes multi-thread 3.8× slower than single-thread on ARM small cores
4. **Eager load + model warmup** — eliminates 5.8s cold-start outlier; warmup uses full few-shot prompt to trigger llama.cpp's internal allocations
5. **Hybrid architecture** — threshold rules filter 82% of windows in <0.1ms; low-severity bypass skips another ~30% of potential LLM calls

## Documentation

- [`submission/SUBMISSION.md`](submission/SUBMISSION.md) — Full Devpost submission write-up with setup instructions
- [`docs/benchmark_results.md`](docs/benchmark_results.md) — Detailed benchmark data and methodology
- [`docs/process_report.md`](docs/process_report.md) — Decision log and implementation notes
- [`docs/rpi5_deployment.md`](docs/rpi5_deployment.md) — RPi 5 deployment guide with systemd service

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
