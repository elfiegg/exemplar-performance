# Muse Spark Open Pre-training with TorchTitan

This recipe runs a Muse Spark Open 816B-A42B pre-training throughput benchmark with TorchTitan and Dist-MoE. It runs the TorchTitan recipe `muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark` for 100 steps.

This is a throughput benchmark only. Forced balanced routing replaces the learned routers and the weights are randomly initialized, so the run does not train a usable model. Master weights and Adam state are BF16, which gives a higher loss than FP32 master weights.

## Supported configurations

- GPUs: VR, four GPUs per node, all in one NVLink domain
- Scale: 64 GPUs
- Precision: BF16 compute, BF16 master weights, gradients, Adam moments, and gradient reductions
- Sequence length: 32768
- Model size: `816b`

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | MBS | GBS | GA  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BF16      |  64  | 32768  |  1  |  1  |  2  | 64  | 32  |  1  | 64  |  2  |

Attention uses FlashAttention 4 varlen kernels, with selective K/V gather in the sliding-window layers. Attention activations are offloaded to host memory, and a 112 GiB Dist-MoE activation slot saves the remaining MoE activations. The data is TorchTitan's `c4_test` fixture, packed into 32K-token rows.

## Prerequisites

- A Slurm cluster with VR nodes and Pyxis/Enroot
- Python 3.12 or conda
- An `HF_TOKEN` that can read `meta-models/Muse-Spark-Open-816B-A42B`
- The environment prepared with the installer described in the [main README](../../../README.md)

The installer clones TorchTitan, Dist-MoE, torch-remat, and attention-gym at the commits in `metadata.yaml`, and downloads the tokenizer at a pinned revision. The `container_setup` task then builds `$LLMB_INSTALL/images/muse-spark-torchtitan.sqsh` from the NGC PyTorch 26.11 image: it installs the wheels in `requirements.lock`, each verified by its sha256 hash, and then the four source checkouts. The image's PyTorch 2.15 predates FSDP2 APIs this recipe needs, so `requirements.lock` pins a PyTorch 2.16 nightly with its CUDA 13.4 libraries, Triton, and the kernel packages. Keep a copy of the nightly wheel, since nightly indexes are pruned.

## Run with `llmb-run`

From `$LLMB_INSTALL`, submit the benchmark:

```bash
llmb-run submit -w pretrain_muse-spark -s 816b --dtype bf16 --scale 64
```

Set `gpu_type` in the selected `llmb-run` cluster configuration to `vr`. See the [`llmb-run` documentation](../../../cli/llmb-run/README.md) for cluster configuration and submission options.

## Run directly

Activate the environment created by the installer, then submit the launch script from the installed recipe directory:

```bash
cd "$LLMB_INSTALL/llmb_repo/muse_spark/pretrain/torchtitan"

LLMB_EXPERIMENT_DIR=<output directory> \
JOB_TOTAL_GPUS=64 \
GPU_TYPE=vr \
sbatch --nodes=16 --account=<account> --partition=<partition> ./launch.sh
```

Required environment variables:

- `LLMB_INSTALL`: top-level LLMB installation directory
- `LLMB_WORKLOAD`: `$LLMB_INSTALL/workloads/pretrain_muse-spark`
- `LLMB_EXPERIMENT_DIR`: directory for logs and TorchTitan output
- `JOB_TOTAL_GPUS`: `64`
- `GPU_TYPE`: `vr`

The launch script sets `NCCL_NVLS_ENABLE=1` for FSDP symmetric memory and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, so pinned host buffers keep their exact size instead of rounding up to a power of two.

## Results

TorchTitan logs one line per step to `log-muse-spark-torchtitan_<jobid>.out`. Report the mean over steps 71-90 of the logged throughput (`tps`, tokens per second per GPU) and step time. The first steps include compilation and CUDA graph capture.

## Known issues

- With FSDP symmetric memory and expert parallelism, NCCL can hang in `destroy_process_group` after training finishes. The results are complete once step 100 is logged; the job then ends at its time limit.
