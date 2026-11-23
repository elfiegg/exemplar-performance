#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the run image: the base image plus the hash-locked wheels in
# requirements.lock and the source checkouts pinned in metadata.yaml.

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

: "${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}"
: "${LLMB_WORKLOAD:?LLMB_WORKLOAD is a required variable.}"
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BASE_SQSH=$LLMB_INSTALL/images/muse-spark-torchtitan-base.sqsh
RUN_SQSH=$LLMB_INSTALL/images/muse-spark-torchtitan.sqsh
# Installed in dependency order.
SOURCES=(remat dist_moe attention-gym torchtitan)

[[ -f $BASE_SQSH ]] || { echo "ERROR: base image not found: $BASE_SQSH" >&2; exit 1; }
for source in "${SOURCES[@]}"; do
    [[ -d $LLMB_WORKLOAD/$source ]] || { echo "ERROR: checkout not found: $LLMB_WORKLOAD/$source" >&2; exit 1; }
done

BUILD_DIR=$(mktemp -d "$LLMB_INSTALL/.cache/muse-spark-image-XXXXXX")
trap 'rm -rf "$BUILD_DIR"' EXIT
export ENROOT_DATA_PATH=$BUILD_DIR/data ENROOT_CACHE_PATH=$BUILD_DIR/cache ENROOT_TEMP_PATH=$BUILD_DIR/tmp
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"

enroot create --name muse-spark-build "$BASE_SQSH"
enroot start --root --rw \
    --mount "$RECIPE_DIR:/opt/recipe" \
    --mount "$LLMB_WORKLOAD:/opt/workload" \
    muse-spark-build \
    bash -c "
        set -eu -o pipefail
        python3 -m pip install --no-deps --require-hashes -r /opt/recipe/requirements.lock
        for source in ${SOURCES[*]}; do
            python3 -m pip install --no-deps --no-build-isolation /opt/workload/\$source
        done
        python3 -m pip freeze > /opt/muse_spark_packages.txt
    "
# enroot start leaves the mount points as empty, root-only directories, which
# enroot export cannot read.
rmdir "$ENROOT_DATA_PATH/muse-spark-build/opt/recipe" "$ENROOT_DATA_PATH/muse-spark-build/opt/workload"
enroot export --output "$RUN_SQSH.partial" muse-spark-build
mv "$RUN_SQSH.partial" "$RUN_SQSH"
echo "Built $RUN_SQSH"
