"""
Parallel wrapper for compute_hcs_only.py

Safely splits the JSONL file into chunks and processes them in parallel
across multiple workers. Each worker uses the original compute_hcs_only.py
logic, ensuring compatibility with GroundingDINO's single-image design.

Usage:
    python compute_hcs_parallel.py --input-jsonl results/out_full.jsonl \
        --output-jsonl results/out_full_hcs.jsonl \
        --num-workers 4 \
        [all other compute_hcs_only.py arguments...]
"""

import json
import os
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import sys
import threading
import time
from tqdm import tqdm

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def split_jsonl(input_file: str, num_chunks: int, temp_dir: str) -> List[str]:
    """Split JSONL file into N chunks for parallel processing."""
    print(f"📂 Splitting {input_file} into {num_chunks} chunks...")
    
    # Read all lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    total = len(lines)
    chunk_size = (total + num_chunks - 1) // num_chunks  # Ceiling division
    
    chunk_files = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        
        chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.jsonl")
        with open(chunk_file, "w", encoding="utf-8") as f:
            for line in lines[start_idx:end_idx]:
                f.write(line + "\n")
        
        chunk_files.append(chunk_file)
        print(f"  ✓ Chunk {i+1}/{num_chunks}: {end_idx - start_idx} samples → {chunk_file}")
    
    return chunk_files


def merge_jsonl(chunk_outputs: List[str], final_output: str):
    """Merge multiple JSONL files into one, preserving order."""
    print(f"🔗 Merging {len(chunk_outputs)} chunks into {final_output}...")
    
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    
    with open(final_output, "w", encoding="utf-8") as out_f:
        for chunk_file in sorted(chunk_outputs):  # Sort to maintain order
            if not os.path.exists(chunk_file):
                print(f"⚠️ Warning: Chunk file {chunk_file} not found, skipping")
                continue
            
            with open(chunk_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
    
    print(f"✅ Merged complete → {final_output}")


def run_worker(worker_id: int, chunk_input: str, chunk_output: str, base_args: List[str], log_file: str = None, gpu_id: int = None):
    """Run a single worker process on a chunk with progress logging and GPU assignment."""
    cmd = [
        sys.executable,
        "compute_hcs_only.py",
        "--input-jsonl", chunk_input,
        "--output-jsonl", chunk_output,
    ] + base_args
    
    # Set CUDA_VISIBLE_DEVICES for this worker to use a specific GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Count samples in input chunk for progress tracking
    try:
        with open(chunk_input, 'r') as f:
            total_samples = sum(1 for _ in f if _.strip())
    except:
        total_samples = 0
    
    # Open log file for this worker
    log_fh = open(log_file, 'w') if log_file else None
    
    try:
        # Start process with real-time output and GPU assignment
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env  # Pass environment with CUDA_VISIBLE_DEVICES
        )
        
        # Thread function to read stdout - suppress all output, only log to file
        def read_output(pipe, prefix, log_fh, wid):
            for line in iter(pipe.readline, ''):
                if line and log_fh:
                    log_fh.write(line)
                    log_fh.flush()
                    # Don't print anything to console - all output goes to log file only
        
        # Start threads to read output
        stdout_thread = threading.Thread(
            target=read_output,
            args=(process.stdout, f"[W{worker_id}]", log_fh, worker_id),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_output,
            args=(process.stderr, f"[W{worker_id} ERR]", log_fh, worker_id),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for threads to finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        if return_code != 0:
            # Only print errors, suppress normal completion messages
            if log_file:
                return (False, f"Check log: {log_file}")
            return (False, "Unknown error")
        
        return (True, None)
        
    except Exception as e:
        return (False, str(e))
    finally:
        if log_fh:
            log_fh.close()


def process_chunk(args_tuple):
    """Process a single chunk (must be at module level for multiprocessing)."""
    idx, chunk_input, temp_dir, base_args, log_file, gpu_id, chunk_output = args_tuple
    success, error = run_worker(idx, chunk_input, chunk_output, base_args, log_file, gpu_id)
    return idx, chunk_output, success, error

def count_processed_samples(output_file: str) -> int:
    """Count how many samples have been processed (have hcs_score)."""
    if not os.path.exists(output_file):
        return 0
    try:
        count = 0
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        if item.get('hcs_score') is not None:
                            count += 1
                    except:
                        pass
        return count
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Parallel HCS computation using multiple workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 4 parallel workers
  python compute_hcs_parallel.py --input-jsonl results/out.jsonl \\
      --output-jsonl results/out_hcs.jsonl \\
      --num-workers 4 \\
      --detector-config models/GroundingDINO/.../config.py \\
      --detector-model-path models/.../model.pth \\
      --images-root datasets/val2017
  
  # Use GPU-specific workers (assign different GPUs)
  CUDA_VISIBLE_DEVICES=0 python compute_hcs_parallel.py ... --num-workers 1 &
  CUDA_VISIBLE_DEVICES=1 python compute_hcs_parallel.py ... --num-workers 1 &
        """
    )
    
    # Required arguments
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file")
    parser.add_argument("--num-workers", type=int, default=None, 
                       help="Number of parallel workers (default: auto-detect = number of GPUs)")
    parser.add_argument("--gpus", type=str, default=None,
                       help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). Default: auto-detect all available GPUs")
    
    # Forward all other compute_hcs_only.py arguments
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--detector-model-path", required=True)
    parser.add_argument("--cache-dir", default="./cache_dir")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--area-threshold", type=float, default=0.001)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-interval", type=int, default=100, 
                       help="Checkpoint interval (only if compute_hcs_only.py supports it)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    
    args = parser.parse_args()
    
    # Detect available GPUs
    available_gpus = []
    if TORCH_AVAILABLE and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        available_gpus = list(range(num_gpus))
        print(f"🎮 Detected {num_gpus} GPU(s): {available_gpus}")
    else:
        # Try using nvidia-smi as fallback
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                available_gpus = list(range(len(gpu_lines)))
                print(f"🎮 Detected {len(available_gpus)} GPU(s) via nvidia-smi: {available_gpus}")
        except:
            print("⚠️ Could not detect GPUs, will use CPU or single GPU")
    
    # Determine which GPUs to use
    if args.gpus:
        # User specified GPUs
        gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
        print(f"🎯 Using user-specified GPUs: {gpu_list}")
    elif available_gpus:
        # Auto-use all available GPUs
        gpu_list = available_gpus
        print(f"🎯 Auto-assigning GPUs: {gpu_list}")
    else:
        # No GPUs detected, all workers use default (CPU or GPU 0)
        gpu_list = [None] * (args.num_workers or 1)
        print(f"⚠️ No GPUs detected, workers will use default device")
    
    # Determine number of workers - 4 workers per GPU for optimal balance
    if args.num_workers is None:
        # Auto-set to 4 workers per GPU
        if gpu_list and len(gpu_list) > 0:
            num_workers = len(gpu_list) * 4  # 4 workers per GPU
            print(f"🔄 Auto-setting num-workers to {num_workers} (4 workers per GPU)")
        else:
            num_workers = 1
            print(f"🔄 Auto-setting num-workers to {num_workers}")
    else:
        num_workers = args.num_workers
    
    # Assign GPUs to workers (round-robin if more workers than GPUs)
    worker_gpu_assignment = []
    for i in range(num_workers):
        if gpu_list:
            gpu_id = gpu_list[i % len(gpu_list)]  # Round-robin assignment
        else:
            gpu_id = None
        worker_gpu_assignment.append(gpu_id)
    
    print(f"\n📋 Worker GPU Assignment:")
    for i, gpu_id in enumerate(worker_gpu_assignment):
        if gpu_id is not None:
            print(f"   Worker {i} → GPU {gpu_id}")
        else:
            print(f"   Worker {i} → Default device")
    
    # Build base arguments to forward
    base_args = [
        "--detector-config", args.detector_config,
        "--detector-model-path", args.detector_model_path,
        "--cache-dir", args.cache_dir,
        "--box-threshold", str(args.box_threshold),
        "--area-threshold", str(args.area_threshold),
        "--images-root", args.images_root,
        "--device", args.device,
    ]
    # Note: Not passing --checkpoint-interval since compute_hcs_only.py will use default (100)
    # If your compute_hcs_only.py has this argument, uncomment the line below:
    # base_args.extend(["--checkpoint-interval", str(args.checkpoint_interval)])
    if args.resume:
        base_args.append("--resume")
    
    # Create temp directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="hcs_parallel_")
    print(f"📁 Temporary directory: {temp_dir}")
    
    try:
        # Step 1: Split input file
        chunk_inputs = split_jsonl(args.input_jsonl, num_workers, temp_dir)
        
        # Step 2: Process chunks in parallel
        chunk_outputs = []
        processes = []
        
        import multiprocessing
        from multiprocessing import Pool
        
        # Create log files for each worker
        log_files = [
            os.path.join(temp_dir, f"worker_{idx:03d}.log")
            for idx in range(len(chunk_inputs))
        ]
        
        # Prepare arguments for each worker (must be picklable)
        chunk_outputs_list = [
            os.path.join(temp_dir, f"output_chunk_{idx:03d}.jsonl")
            for idx in range(len(chunk_inputs))
        ]
        
        worker_args = [
            (idx, chunk_input, temp_dir, base_args, log_file, worker_gpu_assignment[idx], chunk_output)
            for idx, (chunk_input, log_file, chunk_output) in enumerate(zip(chunk_inputs, log_files, chunk_outputs_list))
        ]
        
        print(f"\n🔄 Starting {num_workers} parallel workers...")
        print(f"📝 Logs saved to: {temp_dir}/worker_*.log\n")
        
        # Start workers in background and monitor progress
        pool = Pool(processes=num_workers)
        
        # Track worker outputs and create progress bars
        worker_totals = {}
        progress_bars = {}
        
        for idx in range(num_workers):
            try:
                chunk_size = sum(1 for _ in open(chunk_inputs[idx], 'r') if _.strip())
                worker_totals[idx] = {
                    'total': chunk_size,
                    'output': chunk_outputs_list[idx],
                    'gpu': worker_gpu_assignment[idx]
                }
                # Create separate progress bar for each worker
                gpu_label = f"GPU{worker_gpu_assignment[idx]}" if worker_gpu_assignment[idx] is not None else "Default"
                progress_bars[idx] = tqdm(
                    total=chunk_size, 
                    desc=f"Worker {idx} ({gpu_label})", 
                    position=idx,
                    leave=True,
                    ncols=120,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    unit="samples"
                )
            except Exception as e:
                worker_totals[idx] = {'total': 0, 'output': chunk_outputs_list[idx], 'gpu': worker_gpu_assignment[idx]}
                progress_bars[idx] = None
        
        # Submit all tasks
        async_results = [pool.apply_async(process_chunk, (args,)) for args in worker_args]
        
        # Monitor progress - update each worker's progress bar
        try:
            while True:
                all_done = True
                
                for idx, result in enumerate(async_results):
                    if not result.ready():
                        all_done = False
                        # Update this worker's progress bar
                        if progress_bars[idx] is not None:
                            processed = count_processed_samples(worker_totals[idx]['output'])
                            current_n = progress_bars[idx].n
                            if processed > current_n:
                                progress_bars[idx].update(processed - current_n)
                    else:
                        # Worker completed, final update
                        if progress_bars[idx] is not None:
                            processed = count_processed_samples(worker_totals[idx]['output'])
                            progress_bars[idx].n = processed
                            progress_bars[idx].refresh()
                
                if all_done:
                    break
                    
                time.sleep(1)  # Update every second
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted! Workers may still be running...")
        
        # Close all progress bars
        for pb in progress_bars.values():
            if pb is not None:
                pb.close()
        
        # Get results
        results = [result.get() for result in async_results]
        pool.close()
        pool.join()
        
        # Collect outputs
        chunk_outputs = []
        failed_workers = []
        for idx, chunk_output, success, error in results:
            if success:
                chunk_outputs.append(chunk_output)
            else:
                failed_workers.append((idx, error))
        
        if failed_workers:
            print(f"\n⚠️ Warning: {len(failed_workers)} workers failed:")
            for idx, error in failed_workers:
                print(f"   Worker {idx}: {error}")
            if len(chunk_outputs) == 0:
                print("❌ All workers failed! Exiting.")
                return 1
        
        # Step 3: Merge results
        print(f"\n")
        merge_jsonl(chunk_outputs, args.output_jsonl)
        
        print(f"\n✅ Parallel processing complete!")
        print(f"   Processed {num_workers} chunks")
        print(f"   📁 Final output saved to: {os.path.abspath(args.output_jsonl)}")
        print(f"   💡 To resume if interrupted, use: --resume flag")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✅ Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

