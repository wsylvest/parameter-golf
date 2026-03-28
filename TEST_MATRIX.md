ABOUTME: This file tracks test commands run and their pass/fail status per phase.
ABOUTME: It is used by Claude Code to ensure testing discipline across phases.

# Test Matrix

## All Phases: Syntax Check

| Phase | Command | Status |
|-------|---------|--------|
| 0 | `python3 -c "import py_compile; py_compile.compile('train_gpt.py', doraise=True)"` | PASS |
| 1 | same | PASS |
| 2 | same | PASS |
| 3 | same | PASS |
| 4 | same | PASS |
| 5 | same | PASS |
| 6 | same | PASS |
| 8 | same | PASS |

## Line Count Tracking

| Phase | Lines | Budget Remaining |
|-------|-------|-----------------|
| 0 | 1307 | 193 |
| 1 | 1321 | 179 |
| 2 | 1320 | 180 |
| 3 | 1325 | 175 |
| 4 | 1332 | 168 |
| 5 | 1342 | 158 |
| 6 | 1353 | 147 |
| 8 | 1412 | 88 |

## Pending: GPU Canary

All technique implementations need GPU validation. Recommended canary command:

```bash
RUN_ID=canary_full \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
NUM_LAYERS=11 MLP_MULT=3 \
EMA_DECAY=0.997 \
LN_SCALE=1 \
ROPE_DIMS=16 \
XSA_LAST_N=4 \
QAT_START_FRAC=0.85 \
MUON_WD=0.005 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=512 \
TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
