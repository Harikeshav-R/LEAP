#!/usr/bin/env python3
import argparse
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Dict


def cleanup_process(proc):
    if proc and proc.poll() is None:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait()


def run_benchmark(
        executable: str,
        model: str,
        tokenizer: str,
        transport: str,
        steps: int,
        prompt: str,
        num_workers: int = 1,
        total_layers: int = 32,
        host: str = "127.0.0.1",
        base_port: int = 9999,
        manual: bool = False,
        next_host: str = None,
        next_port: int = None,
        split_layer: int = None
) -> Dict[str, float]:
    """
    Runs a benchmark for a specific transport mode.
    If manual=True, assumes workers are already running and only starts Master.
    """
    # print(f"--- Benchmarking {transport.upper()} Transport ...") # Reduced noise

    # Check for platform compatibility
    if transport == "kernel" and sys.platform == "darwin":
        print("Skipping Kernel transport on macOS (Linux only).")
        return {}

    processes = []

    # Calculate layer splits (only needed for auto-mode or if split_layer not provided)
    if not split_layer:
        # Default even split policy
        total_nodes = num_workers + 1
        layers_per_node = total_layers // total_nodes
        # Master gets first chunk
        master_split = layers_per_node + (1 if total_layers % total_nodes > 0 else 0)
    else:
        master_split = split_layer

    try:
        if not manual:
            # Auto-spawn Workers logic (Local Ring)
            # ... (Existing logic for calculation and spawning)

            # Recalculate full boundaries for auto-spawn
            total_nodes = num_workers + 1
            layers_per_node = total_layers // total_nodes
            remainder = total_layers % total_nodes
            boundaries = [0]
            current = 0
            for i in range(total_nodes):
                extra = 1 if i < remainder else 0
                current += layers_per_node + extra
                boundaries.append(current)

            master_split = boundaries[1]  # Ensure consistent with loop

            for i in range(1, total_nodes):
                node_idx = i
                my_port = base_port + node_idx
                if node_idx == total_nodes - 1:
                    target_port = base_port  # Loop back to Master
                else:
                    target_port = base_port + node_idx + 1

                split_start = boundaries[node_idx]
                split_end = boundaries[node_idx + 1]

                cmd = [
                    executable,
                    model,
                    "--role", "worker",
                    "--split", str(split_start),
                    "--transport", transport,
                    "--host", host,
                    "--port", str(my_port),
                    "--next-host", host,
                    "--next-port", str(target_port),
                ]

                if node_idx < total_nodes - 1:
                    cmd.extend(["--end", str(split_end)])

                if tokenizer:
                    cmd.extend(["--tokenizer", tokenizer])

                print(f"Starting Worker {i}: {' '.join(cmd)}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processes.append(proc)

            # Give workers time to initialize
            time.sleep(2)

            # Auto-mode master targets
            target_host = host
            target_port = base_port + 1

        else:
            # Manual mode master targets
            if not next_host or not next_port:
                print("Error: Manual mode requires --next-host and --next-port")
                return {}
            target_host = next_host
            target_port = next_port

        # Start Master (Node 0)
        master_cmd = [
            executable,
            model,
            "--role", "master",
            "--split", str(master_split),
            "--transport", transport,
            "--host", host,
            "--port", str(base_port),
            "--next-host", target_host,
            "--next-port", str(target_port),
            "-n", str(steps),
        ]

        if tokenizer:
            master_cmd.extend(["--tokenizer", tokenizer])

        master_cmd.extend(["-p", f'"{prompt}"'])

        print(f"Starting Master: {' '.join(master_cmd)}")
        start_time = time.time()

        master_proc = subprocess.run(
            master_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120 + (steps * 0.1)
        )
        end_time = time.time()

        if master_proc.returncode != 0:
            print(f"Master failed. Return code: {master_proc.returncode}")
            print(f"Master Stdout:\n{master_proc.stdout}")
            print(f"Master Stderr:\n{master_proc.stderr}")
            return {}

        output = master_proc.stdout
        error_output = master_proc.stderr

        # Parse output
        tok_s_match = re.search(r"achieved tok/s:\s+([\d\.]+)", error_output)
        if tok_s_match:
            throughput = float(tok_s_match.group(1))
        else:
            throughput = (steps - 1) / (end_time - start_time) if (end_time - start_time) > 0 else 0

        print(f"Benchmark finished. Throughput: {throughput:.2f} tok/s")
        return {"throughput": throughput, "total_time": end_time - start_time}

    except subprocess.TimeoutExpired:
        print("Benchmark timed out!")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    finally:
        # Cleanup all spawned processes (only applicable in auto mode)
        for proc in processes:
            cleanup_process(proc)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LEAP transport modes")
    parser.add_argument("--executable", required=True, help="Path to LEAP inference executable")
    parser.add_argument("--model", required=True, help="Path to .bin model file")
    parser.add_argument("--tokenizer", help="Path to tokenizer.bin file")
    parser.add_argument("--mode", required=True, choices=["tcp", "udp", "kernel"], help="Transport mode to test")
    parser.add_argument("--steps", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog", help="Prompt to use")

    parser.add_argument("--workers", type=int, default=1, help="Number of worker nodes (auto mode only)")
    parser.add_argument("--layers", type=int, default=32, help="Total number of layers in the model")

    # Network config
    parser.add_argument("--host", default="127.0.0.1", help="Master bind IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9999, help="Master bind Port (default: 9999)")

    # Manual distributed mode args
    parser.add_argument("--manual", action="store_true", help="Manual mode: Don't spawn workers, just run Master")
    parser.add_argument("--next-host", help="Next node IP (required for manual mode)")
    parser.add_argument("--next-port", type=int, help="Next node port (required for manual mode)")
    parser.add_argument("--split", type=int, help="Master split layer (override auto-calc)")

    # New argument
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs (default: 10)")

    args = parser.parse_args()

    if not os.path.exists(args.executable):
        print(f"Error: Executable not found at {args.executable}")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    if args.manual and (not args.next_host or not args.next_port):
        print("Error: --manual mode requires --next-host and --next-port")
        return

    throughputs = []
    latencies = []

    print(f"Starting benchmark for mode: {args.mode.upper()}")
    print(f"Configuration: {'Manual Distributed' if args.manual else f'{args.workers} Worker(s) + 1 Master'}")
    print(f"Executing {args.runs} runs...")
    print("-" * 60)

    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs}...", end=" ", flush=True)
        res = run_benchmark(
            args.executable,
            args.model,
            args.tokenizer,
            args.mode,
            args.steps,
            args.prompt,
            num_workers=args.workers,
            total_layers=args.layers,
            host=args.host,
            base_port=args.port,
            manual=args.manual,
            next_host=args.next_host,
            next_port=args.next_port,
            split_layer=args.split
        )

        if res:
            throughputs.append(res['throughput'])
            latencies.append(res['total_time'])
            print(f"Done ({res['throughput']:.2f} tok/s)")
        else:
            print("Failed")

    if not throughputs:
        print("\nNo successful runs.")
        return

    # Calculate Statistics
    def stats(data):
        if not data: return 0, 0, 0, 0, 0, 0
        return (
            statistics.mean(data),
            statistics.median(data),
            min(data),
            max(data),
            statistics.stdev(data) if len(data) > 1 else 0,
            sorted(data)[int(len(data) * 0.95)] if len(data) > 0 else 0  # P95
        )

    t_mean, t_med, t_min, t_max, t_std, t_p95 = stats(throughputs)
    l_mean, l_med, l_min, l_max, l_std, l_p95 = stats(latencies)

    print("\n" + "=" * 60)
    print(f"{'METRIC':<15} | {'MEAN':<10} | {'MEDIAN':<10} | {'MIN':<10} | {'MAX':<10} | {'STD DEV':<10}")
    print("-" * 75)
    print(f"{'Throughput':<15} | {t_mean:<10.2f} | {t_med:<10.2f} | {t_min:<10.2f} | {t_max:<10.2f} | {t_std:<10.2f}")
    print(f"{'Latency (s)':<15} | {l_mean:<10.2f} | {l_med:<10.2f} | {l_min:<10.2f} | {l_max:<10.2f} | {l_std:<10.2f}")
    print("=" * 60)
    print(f"Detailed Latency P95: {l_p95:.2f} s")


if __name__ == "__main__":
    main()
