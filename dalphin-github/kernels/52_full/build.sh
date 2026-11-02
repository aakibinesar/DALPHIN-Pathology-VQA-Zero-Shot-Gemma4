#!/bin/sh
# usage: build.sh <name> <KIND q25|q3|q3b|gemma4|gemma31|gemma31b> <LIMIT 0=all> <model source(s), comma separated> [DTYPE float16|bfloat16] [INTERNET true|false]
#   q3b / gemma4 / gemma31 spread the model over all visible GPUs (device_map=auto); with one GPU they fall back to 4-bit bitsandbytes (needs the package; INTERNET true lets pip install it).
#   gemma31b = full-precision Gemma-4-31B-it (google/gemma-4/transformers/gemma-4-31b-it/1) quantised to 4-bit on load with bitsandbytes (INTERNET true lets pip install it if missing).
#   gemma31 = Gemma-4-31B-it 4-bit compressed-tensors checkpoint (needs the compressed-tensors package -> INTERNET true).
#   env TASKS_ONLY=tissue,behavior: run only these tasks (in this order); default all six. Each finished task is written to submission/<task>.csv, so a watchdog stop keeps the finished ones.
#   env BODY=body_seq.py CASES=0:150: SEQUENTIAL scenario (contextual, text-only history); CASES = first:last case index (0:0 = all). Create body_seq.py first: python make_seq_body.py
#   env TOKBUDGET=560: Gemma-4 image token budget per image (default 280); checked before the model load. Use together with MAXSIDE=1100 to feed ~native-size images.
#   env NOSUFFIX=diagnosis,tissue TASKMAXTOK=diagnosis=128,tissue=64: drop our task instruction for those tasks (organisers prompt only) and set their new-token limits.
#   env BODY=body_hybrid.py HISTJSON=hist.json TASKS_ONLY=behavior,mc,open: HYBRID contextual run (earlier answers embedded from hist.json made by make_hist.py; python make_hybrid_body.py first).
#   env OPEN_NOSUFFIX=1 OPEN_MAXTOK=256: open-answer task only: drop our extra instruction (organisers prompt only) and raise the new-token limit.
#   env LOOKAHEAD=neoplasm=diagnosis (hybrid body only): when generating the first task, the second task's question + answer (from HISTJSON) are given as earlier turns even though that question comes later in the file.
#   env MAXSIDE=1344: longest image side after resizing (default 896); a resolution test checks before the model load that the processor really gives more image tokens.
#   env WATCHDOG=<seconds>: diagnostic - after that long the run prints where it is stuck and exits (so the log is kept). 0 = off.
#   machine_shape is set explicitly to NvidiaTeslaT4 (the T4 x2 machine); push with: kaggle kernels push -p . --accelerator NvidiaTeslaT4
NAME=$1; KIND=$2; LIMIT=$3; MS=$4; DTYPE=${5:-float16}; NET=${6:-false}; WD=${WATCHDOG:-0}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ [ "$WD" != "0" ] && printf 'import faulthandler; faulthandler.dump_traceback_later(%s, exit=True)\n' "$WD"; TO=$(echo "${TASKS_ONLY:-}" | sed 's/[^,][^,]*/"&"/g'); CR=$(echo "${CASES:-0:0}" | sed 's/:/, /'); printf 'import os\nimport numpy as np\nimport pandas as pd\nKIND, LIMIT, DTYPE = "%s", %s, "%s"\nONLY_TASKS = [%s]\nCASE_RANGE = (%s)\n' "$KIND" "$LIMIT" "$DTYPE" "$TO" "$CR"; [ -n "${MAXSIDE:-}" ] && printf 'MAXSIDE_ARG = %s\n' "$MAXSIDE"; [ -n "${TOKBUDGET:-}" ] && printf 'TOKBUDGET_ARG = %s\n' "$TOKBUDGET"; [ -n "${OPEN_MAXTOK:-}" ] && printf 'OPEN_MAXTOK_ARG = %s\n' "$OPEN_MAXTOK"; [ -n "${OPEN_NOSUFFIX:-}" ] && printf 'OPEN_NOSUFFIX_ARG = 1\n'; [ -n "${NOSUFFIX:-}" ] && printf 'NOSUFFIX_ARG = [%s]\n' "$(echo "$NOSUFFIX" | sed 's/[^,][^,]*/"&"/g')"; [ -n "${HISTJSON:-}" ] && printf 'HIST_ANS = %s
' "$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8")))' "$HISTJSON")"; [ -n "${LOOKAHEAD:-}" ] && printf 'LOOKAHEAD_ARG = {"%s": ["%s"]}
' "${LOOKAHEAD%%=*}" "${LOOKAHEAD#*=}"; [ -n "${TASKMAXTOK:-}" ] && printf 'TASKMAXTOK_ARG = {%s}\n' "$(echo "$TASKMAXTOK" | sed 's/\([a-z]*\)=\([0-9]*\)/"\1": \2/g')"; grep -v '^import pandas\|^import numpy' ../../dalphin_pipeline.py | grep -v '^if __name__'; cat "${BODY:-body.py}"; } > full.py
MSJ=$(echo "$MS" | sed 's/[^,]*/"&"/g')
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/dalphin-52-$NAME", "title": "dalphin-52-$NAME", "code_file": "full.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": $NET, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": [], "competition_sources": [], "kernel_sources": ["$KUSER/dalphin-51-images"], "model_sources": [$MSJ]}
JSON
