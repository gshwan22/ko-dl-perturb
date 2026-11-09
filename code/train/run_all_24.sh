#!/bin/bash
# 24 runs queue: 4 conditions × 6 models, sequential, resumable
set +e  # don't exit on individual run failure
cd ~/DL-topology

VENV=~/DL-paper/venv_fresh
SCRIPT=H/scripts/train_sft_lora_v2.py
MODELS_DIR=~/DL-paper/models
DATA_DIR=H/data
RUNS_DIR=H/runs
LOGS_DIR=H/logs
VAL=$DATA_DIR/val.jsonl

mkdir -p $RUNS_DIR $LOGS_DIR

# 4 conditions
CONDS=("C0_original" "C1_gpt_dlmin" "C2_algo_dlmin" "C3_algo_dlmax")

# 6 models in priority order: 4B first (fast, finish before move), then 8B
MODELS_4B=("gemma3-4b" "skt-ax4-light")
MODELS_8B=("qwen3-8b" "llama31-8b" "mistral-7b" "exaone35-7.8b")

log() { echo "[$(date '+%F %T')] $1" | tee -a $LOGS_DIR/master_24.log; }

log "==================== 24 RUNS START ===================="
log "Order: 4B models first (gemma3, skt-ax4), then 8B (qwen, llama, mistral, exaone)"
log "Subsample: 30K per condition"
log "Pilot done: C0 gemma3-4b (will skip)"

run_one() {
    local model=$1
    local cond=$2
    local common_args=$3
    local trust=$4
    
    local outdir=$RUNS_DIR/${cond}_${model}
    local train=$DATA_DIR/${cond}_30k.train.jsonl
    local logf=$LOGS_DIR/${cond}_${model}.log
    
    if [ -f "$outdir/adapter_config.json" ]; then
        log "  ⏭️  SKIP ${cond}/${model} (already done)"
        return 0
    fi
    
    log "  ▶ START ${cond}/${model}"
    local t0=$(date +%s)
    
    $VENV/bin/python $SCRIPT \
        --model_name_or_path "$MODELS_DIR/$model" \
        --train_file "$train" \
        --eval_file "$VAL" \
        --output_dir "$outdir" \
        --trust_remote_code "$trust" \
        --max_length 2048 \
        --lr 2e-4 \
        --epochs 1 \
        $common_args \
        --lora_r 16 --lora_alpha 32 \
        --logging_steps 50 \
        --save_steps 500 \
        --save_total_limit 2 \
        > "$logf" 2>&1
    
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$(( (t1 - t0) / 60 ))
    
    if [ -f "$outdir/adapter_config.json" ]; then
        log "  ✅ DONE ${cond}/${model} (${elapsed}min)"
    else
        log "  ❌ FAIL ${cond}/${model} (rc=$rc, ${elapsed}min)"
    fi
}

# 4B common settings
COMMON_4B="--per_device_train_batch_size 8 --grad_accum 2"
# 8B common settings (smaller batch for memory)
COMMON_8B="--per_device_train_batch_size 4 --grad_accum 4"

# ============ 4B models first (~30h total) ============
log "===== 4B models (gemma3-4b, skt-ax4-light) ====="
for model in "${MODELS_4B[@]}"; do
    for cond in "${CONDS[@]}"; do
        run_one "$model" "$cond" "$COMMON_4B" "false"
    done
done

# ============ 8B models (~112h total) ============
log "===== 8B models (qwen3, llama31, mistral, exaone35) ====="

# Non-exaone (no trust_remote_code)
for model in qwen3-8b llama31-8b mistral-7b; do
    for cond in "${CONDS[@]}"; do
        run_one "$model" "$cond" "$COMMON_8B" "false"
    done
done

# EXAONE (trust_remote_code=true)
for cond in "${CONDS[@]}"; do
    run_one "exaone35-7.8b" "$cond" "$COMMON_8B" "true"
done

log "==================== 24 RUNS QUEUE COMPLETE ===================="
