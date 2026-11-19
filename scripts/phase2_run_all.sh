#!/bin/sh
# Runs every Phase 2 test experiment as an independent process (JOBS at a time). ~5-8 min per job on CPU.
#   sh scripts/phase2_run_all.sh            (PY=python JOBS=5)
set -e
PY="${PY:-python}"; JOBS="${JOBS:-5}"; CFG=configs/phase2/m5_phase2.yaml
mkdir -p outputs/phase2/logs
{
  for s in 42 43 44; do for m in deepar_v2 hybrid deepar_v2_nomask deepar_p1; do echo "$m $s"; done; done
  for m in hyb_zi_naive hyb_zi_inflate hyb_gated hyb_none hyb_no_persist hyb_no_dl hyb_nomask hyb_p1 \
           dar_v2_no_val dar_v2_age dar_v2_age_adamw; do echo "$m 42"; done
} | xargs -P "$JOBS" -L 1 sh -c '
  d=outputs/phase2/runs/$0_s$1
  if [ -f "$d/forecasts.parquet" ]; then echo "skip $d"; exit 0; fi
  '"$PY"' -m forecastkit.cli run '"$CFG"' --models $0 --set seed=$1 --set run_name=phase2/runs/$0_s$1 \
     > outputs/phase2/logs/run_$0_s$1.log 2>&1 && echo "done $d" || echo "FAILED $d"'
