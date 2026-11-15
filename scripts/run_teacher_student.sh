#!/bin/bash
# Train the teacher-student sweeps and compute behavioral recovery ranks
# One sweep per setting in SETTINGS; per sweep, all
# (seed, student width d, probe set, reference point) combinations.
#
#   bash scripts/run_teacher_student.sh                 # all five settings
#   bash scripts/run_teacher_student.sh in10_out10      # one setting
#
# Output: $RESULTS_DIR/ts_<setting>/in<i>_th10_out<o>_sd<d>_bias_seed<s>/
#   model/checkpoint.pt       trained student, teacher, data, 40 training snapshots
#   model/eigen_cache/        behavioral eigenpairs, reused across the four runs below
#   runs/topm_-1_probesrc_<train|test>_pvec_<absolute|displacement>/
#                             projection curves (experiment_outputs.pt, summary.json)
#   temporal/                 (in10_out10 only) recovery rank at the 40 training
#                             snapshots, recovery_ranks_long.csv and results.json
#
# pvec_absolute is theta_ref = 0; pvec_displacement is theta_ref = theta_0.
# Runs are CPU-only, float64; a single run takes seconds to a few minutes.
# The snapshot ranks of the main setting are computed at the end, without
# retraining, TEMPORAL_JOBS runs in parallel with TEMPORAL_THREADS BLAS threads each.


set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS_DIR=${RESULTS_DIR:-results}
TEMPORAL_JOBS=${TEMPORAL_JOBS:-8}
TEMPORAL_THREADS=${TEMPORAL_THREADS:-4}

# setting name : input dim : output dim : label-noise std
ALL_SETTINGS=(in10_out10:10:10:0        # main setting
              in20_out10:20:10:0        # wider input
              in10_out20:10:20:0        # wider output
              in20_out5:20:5:0          # wide input, narrow output
              in10_out10_noise:10:10:0.1) # label noise, sigma^2 = 0.01

SEEDS=(0 1 2 3 4)
WIDTHS=$(seq 3 20)
TEACHER_WIDTH=10

if [[ $# -gt 0 ]]; then
    SETTINGS=()
    for want in "$@"; do
        for s in "${ALL_SETTINGS[@]}"; do [[ ${s%%:*} == "$want" ]] && SETTINGS+=("$s"); done
    done
else
    SETTINGS=("${ALL_SETTINGS[@]}")
fi

for s in "${SETTINGS[@]}"; do
    IFS=: read -r name din dout noise <<< "$s"
    root="$RESULTS_DIR/ts_$name"
    for seed in "${SEEDS[@]}"; do
    for d in $WIDTHS; do
        cfg="$root/in${din}_th${TEACHER_WIDTH}_out${dout}_sd${d}_bias_seed${seed}"
        for probe in train test; do
        for pvec in displacement absolute; do
            sub="runs/topm_-1_probesrc_${probe}_pvec_${pvec}"
            if [[ -f "$cfg/$sub/summary.json" ]]; then continue; fi
            echo "[$(date +%H:%M:%S)] $name seed=$seed d=$d probe=$probe pvec=$pvec"
            python src/run_toy_teacher_student.py \
                --seed "$seed" \
                --input-dim "$din" --output-dim "$dout" \
                --teacher-hidden-dim "$TEACHER_WIDTH" --hidden-dim "$d" \
                --noise-std "$noise" \
                --train-size 2048 --test-size 512 \
                --steps 4000 --lr 1e-3 \
                --top-m -1 \
                --probe-source "$probe" --projection-vector-mode "$pvec" \
                --out-dir "$cfg" --run-subdir-name "$sub"
        done
        done
    done
    done
done

# Recovery rank at the training snapshots of the main setting (no retraining).
for s in "${SETTINGS[@]}"; do
    [[ ${s%%:*} == in10_out10 ]] || continue
    echo "[$(date +%H:%M:%S)] in10_out10 snapshot ranks"
    find "$RESULTS_DIR/ts_in10_out10" -mindepth 1 -maxdepth 1 -type d -name 'in*_seed*' | sort \
        | while read -r d; do [[ -f "$d/temporal/recovery_ranks_long.csv" ]] || echo "$d"; done \
        | OMP_NUM_THREADS=$TEMPORAL_THREADS MKL_NUM_THREADS=$TEMPORAL_THREADS xargs -P "$TEMPORAL_JOBS" -I{} \
            sh -c 'python src/run_temporal_recovery.py --run-dir "{}" > "{}/temporal.log" 2>&1 || echo "FAILED {}"'
done
