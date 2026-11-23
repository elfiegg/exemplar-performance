#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=00:30:00

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

: "${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}"
: "${LLMB_WORKLOAD:?LLMB_WORKLOAD is a required variable.}"
: "${LLMB_EXPERIMENT_DIR:?LLMB_EXPERIMENT_DIR is a required variable.}"
: "${GPU_TYPE:?GPU_TYPE is a required variable.}"
: "${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}"

GPU_TYPE=${GPU_TYPE,,}
DTYPE=${DTYPE:-bf16}
DTYPE=${DTYPE,,}

if [[ $GPU_TYPE != "vr" ]]; then
    echo "ERROR: Muse Spark TorchTitan supports VR only." >&2
    exit 1
fi
if [[ $DTYPE != "bf16" ]]; then
    echo "ERROR: Muse Spark TorchTitan supports DTYPE=bf16 only." >&2
    exit 1
fi
if [[ $JOB_TOTAL_GPUS -ne 64 ]]; then
    echo "ERROR: Muse Spark TorchTitan is validated at 64 GPUs." >&2
    exit 1
fi

GPUS_PER_NODE=4
NUM_NODES=$((JOB_TOTAL_GPUS / GPUS_PER_NODE))
IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/muse-spark-torchtitan.sqsh}
TORCHTITAN=$LLMB_WORKLOAD/torchtitan
TOKENIZER=$LLMB_WORKLOAD/Muse-Spark-Open-816B-A42B
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

# The recipe reads the c4_test fixture and the tokenizer relative to the
# TorchTitan checkout.
CONTAINER_MOUNTS="$TORCHTITAN:$TORCHTITAN,$TOKENIZER:$TORCHTITAN/assets/hf/Muse-Spark-Open-816B-A42B:ro,$LLMB_EXPERIMENT_DIR:$LLMB_EXPERIMENT_DIR"
if [[ -n ${RUN_CONF_MOUNTS:-} ]]; then
    CONTAINER_MOUNTS+=",${RUN_CONF_MOUNTS}"
fi

# FSDP symmetric memory uses NVLS. Pinned host buffers above 1 MiB keep their
# exact size instead of a power of two, which activation offload needs to fit.
CONTAINER_ENV=NCCL_NVLS_ENABLE,PYTORCH_ALLOC_CONF
export NCCL_NVLS_ENABLE=1
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1
if [[ -n ${LLMB_CONTAINER_ENV:-} ]]; then
    CONTAINER_ENV+=",${LLMB_CONTAINER_ENV}"
fi

srun --nodes="$NUM_NODES" --ntasks-per-node=1 \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-env="$CONTAINER_ENV" \
    --container-workdir="$TORCHTITAN" \
    --no-container-mount-home \
    --output="$LLMB_EXPERIMENT_DIR/log-muse-spark-torchtitan_%j.out" \
    bash -c "torchrun \
        --nnodes=$NUM_NODES --nproc-per-node=$GPUS_PER_NODE \
        --node-rank=\$SLURM_NODEID \
        --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
        -m torchtitan.train \
        --module torchtitan_recipes.models.muse_spark \
        --config muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark \
        --output-dir $LLMB_EXPERIMENT_DIR/trainer"
