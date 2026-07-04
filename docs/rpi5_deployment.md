# Raspberry Pi 5 Deployment Guide

## Prerequisites

- Raspberry Pi 5 (4GB or 8GB recommended)
- microSD card (32GB+ recommended)
- 64-bit OS (Raspberry Pi OS Lite or Ubuntu 24.04 LTS for ARM64)
- Optional: GPIO-connected BUZZER for hardware alerts

## Step 1 — Install Dependencies

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake git python3 python3-pip python3-venv \
    libopenblas-dev

# Clone the project
git clone <repo-url> /home/pi/arm-optimize-iot
cd /home/pi/arm-optimize-iot
```

## Step 2 — Build llama.cpp for ARM64

```bash
./setup.sh
```

Note: the setup script builds llama.cpp with NEON optimizations (auto-detected on ARM64).
CUDA is disabled (no GPU on RPi5).

## Step 3 — Transfer the Model

On your development machine (after quantization):

```bash
# From dev machine:
scp models/qwen2-0.5b-q4_k_m.gguf pi@<rpi-ip>:/home/pi/arm-optimize-iot/models/
```

Or download and quantize directly on the RPi5:

```bash
cd /home/pi/arm-optimize-iot
python -m src.model_optimization.download_model
python -m src.model_optimization.export_to_gguf
python -m src.model_optimization.quantize_model
```

## Step 4 — Generate Sensor Data

```bash
cd /home/pi/arm-optimize-iot
python -m src.data_processing.sensor_generator --hours 24
```

## Step 5 — Run the Alert Engine

### Direct mode

```bash
python -m src.alerting.alert_engine
```

### As a systemd service (continuous monitoring)

Create `/etc/systemd/system/sensor-alert.service`:

```ini
[Unit]
Description=ARM AI Sensor Triage Alert Engine
After=network.target

[Service]
ExecStart=/home/pi/arm-optimize-iot/venv/bin/python \
    -m src.alerting.alert_engine
WorkingDirectory=/home/pi/arm-optimize-iot
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable sensor-alert.service
sudo systemctl start sensor-alert.service
sudo systemctl status sensor-alert.service
```

## Step 6 — GPIO BUZZER (Optional)

Wire a piezo buzzer to GPIO pin 18 (or configure in `alert_actions.py`).

### Wiring

```
RPi5 GPIO 18  -->  BUZZER (+) 
RPi5 GND       -->  BUZZER (-)
```

### Enable GPIO in alert_actions.py

The `AlertLogger` class in `src/alerting/alert_actions.py` currently logs only.
Extend it to toggle a GPIO pin when alerts fire:

```python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)  # on alert
time.sleep(0.5)
GPIO.output(18, GPIO.LOW)
```

## Step 7 — Benchmark on RPi5

```bash
cd /home/pi/arm-optimize-iot
./benchmark.sh
```

Compare results with the table in `docs/benchmark_results.md`.

## Performance Expectations

| Model          | Quantization | RPi5 Tokens/s | TTFT    | Peak RAM |
|----------------|-------------|---------------|---------|----------|
| Qwen2-0.5B     | Q4_K_M      | ~15-25 t/s    | ~100ms  | ~700 MB  |
| Qwen2-0.5B     | Q8_0        | ~10-18 t/s    | ~150ms  | ~900 MB  |

Expected end-to-end alert latency on RPi5:

| Path                | Latency        |
|---------------------|----------------|
| Threshold only      | < 10 ms        |
| Threshold + LLM     | 200–800 ms     |
| p99 all windows     | < 2000 ms      |

## Troubleshooting

- **Out of memory**: Reduce `n_ctx` in `llm_triage.py` (e.g., 256 instead of 512).
- **Slow inference**: Enable `LLAMA_NATIVE=OFF` and set `LLAMA_ARM64=ON` when building.
- **GPIO permission denied**: Add user to `gpio` group: `sudo usermod -a -G gpio pi`.
