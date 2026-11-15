#!/bin/bash
# Train the width-scaled MNIST MLPs 784 -> w -> w -> 10 and compute the
# projection curves on the leading 12,000 behavioral eigendirections
# Needs a GPU; MNIST is downloaded to data/ on first use.
#
#   bash scripts/run_mnist.sh            # all widths
#   bash scripts/run_mnist.sh 32 64      # selected widths
#
# Output: $MNIST_DIR/width_<w>_<w>_bs_64/model/          checkpoint + eigen cache
#         $MNIST_DIR/width_<w>_<w>_bs_64_topm_<m>_curv_behavioral_pmode_both_probesrc_train/
#                                                         projection curves, theta_ref = theta_0
#
# The eigenpairs are built up in steps of 1,000 (TOPMS): each run reuses the
# cached eigenpairs of the previous one and adds the next 1,000 by Lanczos on
# the deflated operator (4,000 Lanczos steps each). A single run with
# --top-m 12000 would instead keep 48,000 Lanczos vectors of dimension
# 10,000 x 10 in GPU memory (about 19 GB, twice that while stacking them).
# Only the final top-m 12000 directory is used by the figures; the intermediate runs also
# write their own projection curves, which costs extra time.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS_DIR=${RESULTS_DIR:-results}
MNIST_DIR=${MNIST_DIR:-$RESULTS_DIR/mnist}
TOPMS=${TOPMS:-$(seq -s ' ' 1000 1000 12000)}
WIDTHS=("$@")
[[ ${#WIDTHS[@]} -eq 0 ]] && WIDTHS=(32 64 96 128 160)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for w in "${WIDTHS[@]}"; do
for m in $TOPMS; do
    out="$MNIST_DIR/width_${w}_${w}_bs_64_topm_${m}_curv_behavioral_pmode_both_probesrc_train"
    if [[ -f "$out/summary.json" ]]; then continue; fi
    echo "[$(date +%H:%M:%S)] width $w, top-m $m"
    python src/run_mnist_mlp_lazy.py \
        --seed 0 \
        --hidden-dims "$w,$w" \
        --train-batch-size 64 \
        --steps 2000 --lr 1e-3 \
        --probe-source train --probe-size 10000 \
        --top-m "$m" \
        --device cuda \
        --data-root data \
        --out-dir "$MNIST_DIR"
done
done
