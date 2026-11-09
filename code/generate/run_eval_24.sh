#!/bin/bash
LOG=~/DL-topology/H/logs/eval_24.log
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG; }

EVAL_SCRIPT=~/DL-paper/scripts/evaluate_model.py
TEST=~/DL-paper/data/original/sft_side_p0.0.test.sampled.jsonl
PRED_DIR=~/DL-topology/H/eval/predictions
MET_DIR=~/DL-topology/H/eval/metrics

get_venv() {
    case "$1" in
        exaone35-7.8b) echo ~/DL-paper/venv_old/bin/python3 ;;
        *) echo ~/DL-paper/venv_fresh/bin/python3 ;;
    esac
}
get_trust() {
    case "$1" in
        exaone35-7.8b|skt-ax4-light) echo true ;;
        *) echo false ;;
    esac
}

log "▶ Waiting for 24/24 training to finish..."
while true; do
    DONE=$(ls ~/DL-topology/H/runs/*/adapter_config.json 2>/dev/null | wc -l)
    if [ "$DONE" -eq 24 ]; then
        log "✅ 24/24 training complete!"
        break
    fi
    log "  ($DONE/24 done, waiting 10min...)"
    sleep 600
done

MODELS="gemma3-4b llama31-8b mistral-7b qwen3-8b skt-ax4-light exaone35-7.8b"
CONDS="C0_original C1_gpt_dlmin C2_algo_dlmin C3_algo_dlmax"

TOTAL=0; DONE_E=0; FAIL_E=0
for COND in $CONDS; do
  for M in $MODELS; do
    TOTAL=$((TOTAL+1))
    RUN_DIR=~/DL-topology/H/runs/${COND}_${M}
    PRED_FILE=$PRED_DIR/${COND}_${M}.jsonl
    
    if [ -f $PRED_FILE ] && [ $(wc -l < $PRED_FILE) -ge 1000 ]; then
        log "⏭️  Skip ${COND}/${M}"
        DONE_E=$((DONE_E+1))
        continue
    fi
    
    log "▶ EVAL ${COND}/${M}"
    T0=$(date +%s)
    
    $(get_venv $M) $EVAL_SCRIPT \
        --base_model_path ~/DL-paper/models/$M \
        --adapter_path $RUN_DIR \
        --test_data $TEST \
        --output_file $PRED_FILE \
        --max_new_tokens 512 \
        --trust_remote_code $(get_trust $M) \
        > ~/DL-topology/H/logs/eval_${COND}_${M}.log 2>&1
    
    RC=$?
    MIN=$(( ($(date +%s)-T0)/60 ))
    
    if [ $RC -eq 0 ] && [ -f $PRED_FILE ]; then
        log "  ✅ ${MIN}min"
        DONE_E=$((DONE_E+1))
    else
        log "  ❌ rc=$RC (${MIN}min)"
        FAIL_E=$((FAIL_E+1))
    fi
  done
done

log "===== EVAL DONE: $DONE_E/$TOTAL (fail $FAIL_E) ====="

log "▶ Computing metrics..."
~/DL-paper/venv_fresh/bin/python3 ~/DL-paper/scripts/compute_metrics.py \
    --eval_dir $PRED_DIR --output_dir $MET_DIR >> $LOG 2>&1
log "✅ ALL COMPLETE"
