# Benchmarking LEAP

This guide explains how to benchmark the performance (throughput and latency) of the LEAP inference engine across different transport modes (TCP, UDP, Kernel).

## Overview

The `scripts/benchmark.py` script automates the process of setting up a distributed inference ring (currently configured for a 2-node setup: Master + 1 Worker) and measuring performance.

## Usage

```bash
python3 scripts/benchmark.py \
    --executable <path_to_inference_binary> \
    --model <path_to_model.bin> \
    --mode <tcp|udp|kernel> \
    [--tokenizer <path_to_tokenizer.bin>] \
    [--steps 100] \
    [--workers 1] \
    [--layers 32] \
    [--manual] \
    [--next-host <IP>] \
    [--next-port <PORT>] \
    [--split <LAYER>] \
    [--runs 10]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--executable` | Path to the compiled `inference` binary. | Required |
| `--model` | Path to the `.bin` model file. | Required |
| `--mode` | Transport mode to test (`tcp`, `udp`, `kernel`). | Required |
| `--tokenizer` | Path to the `tokenizer.bin` file. | Optional |
| `--steps` | Number of tokens to generate for measurement. | `100` |
| `--prompt` | Input prompt to use. | "The quick brown..." |
| `--workers` | Number of worker nodes to spawn (in addition to Master). | `1` |
| `--layers` | Total number of layers in the model (used for splitting). | `32` |
| `--manual` | **Manual Mode**: Do not spawn workers, only run Master. | `False` |
| `--next-host` | IP of the next node in the ring (required for manual mode). | — |
| `--next-port` | Port of the next node in the ring. | — |
| `--split` | Layer index where Master stops and Worker starts. | — |
| `--runs` | Number of benchmark iterations to run. | `10` |

### Examples

**Run TCP mode with 100 iterations:**

```bash
python3 scripts/benchmark.py \
    --executable ./cmake-build-release/src/inference/inference \
    --model models/llama3-8b.bin \
    --mode tcp \
    --runs 100
```


**Run UDP mode with 3 workers:**

```bash
python3 scripts/benchmark.py \
    --executable ./cmake-build-release/src/inference/inference \
    --model models/llama3-8b.bin \
    --mode udp \
    --workers 3 \
    --layers 32
```


**Manual Distributed Mode (Run on Master Node):**

Run this on the Master node after starting the Worker node on `192.168.1.100:9999`.

```bash
python3 scripts/benchmark.py \
    --executable ./cmake-build-release/src/inference/inference \
    --model models/llama3-8b.bin \
    --manual \
    --next-host 192.168.1.100 \
    --next-port 9999 \
    --split 16
```

## Interpreting Results

The script runs the benchmark `N` times (specified by `--runs`) and outputs detailed statistics:

```
Starting benchmark for mode: UDP
Configuration: 1 Worker(s) + 1 Master
Executing 100 runs...
------------------------------------------------------------
Run 1/100... Done (48.12 tok/s)
...
Run 100/100... Done (49.05 tok/s)

============================================================
METRIC          | MEAN       | MEDIAN     | MIN        | MAX        | STD DEV   
---------------------------------------------------------------------------
Throughput (tok/s) | 48.50      | 48.45      | 45.20      | 51.10      | 1.25      
Latency (s)     | 2.06       | 2.07       | 1.95       | 2.21       | 0.05      
============================================================
Detailed Latency P95: 2.15 s
```

-   **Throughput**: Tokens per second (tok/s) (Higher is better).
-   **Latency**: Total time to generate the requested number of tokens (seconds) (Lower is better).
-   **Std Dev**: Lower indicates more stable performance.

## Notes

-   **Kernel Mode**: Only works on Linux with the `leap_kmod` module loaded. The script will automatically skip Kernel mode if running on macOS.
-   **Architecture**: The benchmark presently assumes a 2-node setup (Master + 1 Worker). Modification to `benchmark.py` is required for larger ring sizes.
