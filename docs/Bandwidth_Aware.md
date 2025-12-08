# Bandwidth-aware migration quick guide

This setup enables GPU P2P profiling and uses the profiling data to steer migrations.

## Start the server (bandwidth-aware)

From repo root, with the project venv activated:

```
python -m llumnix.entrypoints.vllm.api_server \
  --config-file configs/vllm.yml \
  --host 0.0.0.0 --port 8000 \
  --enable-routine-migration \
  --enable-gpu-p2p-profiling \
  --gpu-p2p-min-blocks 1 \
  --gpu-p2p-max-blocks 100 \
  --gpu-p2p-num-samples 10 \
  --gpu-p2p-warmup-blocks 1 \
  --gpu-p2p-max-transfer-time 0.1 \
  --worker-use-ray
```

What it does:
- Runs the standard vLLM entrypoint with routine migration enabled.
- Profiles GPU peer-to-peer bandwidth (per instance) and feeds the profiler into the migration scheduler for bandwidth-aware pairing.

## Run a benchmark (example)

```
python benchmark/benchmark_serving.py \
  --tokenizer gpt2 \
  --backend vLLM \
  --ip_ports 127.0.0.1:8000 \
  --dataset_type sharegpt \
  --dataset_path data/sharegpt_gpt4_small.json \
  --random_prompt_count 30 \
  --max_request_len 2048 \
  --distribution burst \
  --qps 10 \
  --log_filename benchmark/bw_qps10_burst.log
```

Outputs:
- `benchmark/bw_qps10_burst_latency.png`, `..._len.png`, and `..._latency_info.json` for inspection.

## Baseline run for comparison

Start the server without bandwidth profiling (omit the `--enable-gpu-p2p-profiling` flags), then run the same benchmark command with a different `--log_filename`. This gives an apples-to-apples baseline vs bandwidth-aware comparison.

## How it works (brief)

- Each instance profiles GPU P2P transfer time to every other instance for a range of block sizes (log-spaced). Warmup avoids first-touch noise.
- The manager aggregates these measurements, builds interpolation functions, and marks profiling complete.
- The migration scheduler uses a bandwidth-aware pair filter: it only pairs source/destination instances whose estimated transfer time (for the request’s block size) is acceptable, using the optional `--gpu-p2p-max-transfer-time` threshold as a guardrail.
- If profiling is unavailable or yields no data, migration falls back to load-based filtering (no hard failure).

