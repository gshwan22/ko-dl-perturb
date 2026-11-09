#!/bin/bash
set +e
cd ~/DL-topology
VENV=~/DL-paper/venv_fresh; SCRIPT=H/scripts/train_sft_lora_v2.py
MODELS_DIR=~/DL-paper/models; DATA_DIR=H/data; RUNS_DIR=H/runs; LOGS_DIR=H/logs
VAL=$DATA_DIR/val.jsonl; COND=C4_algo_dlneu
log() { echo "[$(date '+%F %T')] $1" | tee -a $LOGS_DIR/master_c4.log; }
run_one() {  # model common_args trust
  local model=$1 common_args=$2 trust=$3
  local outdir=$RUNS_DIR/${COND}_${model} train=$DATA_DIR/${COND}_30k.train.jsonl logf=$LOGS_DIR/${COND}_${model}.log
  [ -f "$outdir/adapter_config.json" ] && { log "SKIP $model"; return; }
  log "START $model"; local t0=$(date +%s)
  $VENV/bin/python $SCRIPT --model_name_or_path "$MODELS_DIR/$model" --train_file "$train" --eval_file "$VAL" \
    --output_dir "$outdir" --trust_remote_code "$trust" --max_length 2048 --lr 2e-4 --epochs 1 $common_args \
    --lora_r 16 --lora_alpha 32 --logging_steps 50 --save_steps 500 --save_total_limit 2 > "$logf" 2>&1
  local el=$(( ($(date +%s)-t0)/60 ))
  [ -f "$outdir/adapter_config.json" ] && log "DONE $model (${el}min)" || log "FAIL $model (${el}min) see $logf"
}
log "==== C4_algo_dlneu 30k: 5 models ===="
run_one gemma3-4b     "--per_device_train_batch_size 8 --grad_accum 2" false
run_one skt-ax4-light "--per_device_train_batch_size 8 --grad_accum 2" true
run_one qwen3-8b      "--per_device_train_batch_size 4 --grad_accum 4" false
run_one llama31-8b    "--per_device_train_batch_size 4 --grad_accum 4" false
run_one mistral-7b    "--per_device_train_batch_size 4 --grad_accum 4" false
log "==== ALL DONE ===="
