#!/usr/bin/env bash
set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/pmmr_posthoc_queue logs/pmmr_posthoc
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

priority() {
  local split="$1"
  case "$split" in
    true_temporal) echo 0 ;;
    unseen_target) echo 1 ;;
    similarity_aware_unseen_target) echo 2 ;;
    all_unseen) echo 3 ;;
    random) echo 4 ;;
    unseen_drug) echo 5 ;;
    temporal_proxy) echo 6 ;;
    *) echo 9 ;;
  esac
}

scan_tasks() {
  find reports/deployment_upgrade_experiments/pmmr_training -name summary.json | while read -r summary; do
    dataset="$(echo "$summary" | awk -F/ '{print $(NF-2)}')"
    splitseed="$(basename "$(dirname "$summary")")"
    split="${splitseed%_seed*}"
    seed="${splitseed##*_seed}"
    run_name="pmmr_${dataset}_${split}_seed${seed}"
    metrics="artifacts/external_runs/pmmr/runs/${run_name}/posthoc_selector/${run_name}_posthoc_metrics.json"
    printf "%s %s %s %s\n" "$(priority "$split")" "$dataset" "$split" "$seed"
  done | sort -n -k1,1 -k2,2 -k3,3 -k4,4
}

posthoc_metrics_path() {
  local dataset="$1" split="$2" seed="$3"
  local run_name="pmmr_${dataset}_${split}_seed${seed}"
  echo "artifacts/external_runs/pmmr/runs/${run_name}/posthoc_selector/${run_name}_posthoc_metrics.json"
}

is_running_task() {
  local run_name="$1"
  pgrep -af "run_posthoc_selector.py" | grep -F -- "--run-name ${run_name}" >/dev/null 2>&1
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1 >= 6 && $1 <= 7 && $2 < 500 {print $1; exit}'
}

launch_task() {
  local gpu="$1" dataset="$2" split="$3" seed="$4"
  local run_name="pmmr_${dataset}_${split}_seed${seed}"
  local log="logs/pmmr_posthoc/${run_name}.log"
  echo "$(date '+%F %T') LAUNCH gpu=${gpu} ${run_name}" | tee -a logs/pmmr_posthoc_queue/queue.log
  nohup bash -lc "export CUDA_VISIBLE_DEVICES=${gpu}; ./.venv/bin/python scripts/run_posthoc_selector.py --workspace . --run-name ${run_name} --regressor-type knn --feature-set enriched9 --accelerator gpu --batch-size 128 --num-workers 0 --num-mc-samples 1 > ${log} 2>&1" >/dev/null 2>&1 &
}

echo "$(date '+%F %T') PMMR posthoc queue daemon started" | tee -a logs/pmmr_posthoc_queue/queue.log
while true; do
  launched=0
  while read -r _priority dataset split seed; do
    [ -f "$(posthoc_metrics_path "$dataset" "$split" "$seed")" ] && continue
    run_name="pmmr_${dataset}_${split}_seed${seed}"
    is_running_task "$run_name" && continue
    gpu="$(free_gpu || true)"
    [ -n "$gpu" ] || break
    launch_task "$gpu" "$dataset" "$split" "$seed"
    launched=1
    sleep 5
  done < <(scan_tasks)

  remaining=0
  while read -r _priority dataset split seed; do
    [ -f "$(posthoc_metrics_path "$dataset" "$split" "$seed")" ] && continue
    remaining=$((remaining + 1))
  done < <(scan_tasks)

  if [ "$remaining" -eq 0 ]; then
    echo "$(date '+%F %T') PMMR posthoc queue finished all available tasks" | tee -a logs/pmmr_posthoc_queue/queue.log
    exit 0
  fi

  if [ "$launched" -eq 0 ]; then
    sleep 60
  fi
done

