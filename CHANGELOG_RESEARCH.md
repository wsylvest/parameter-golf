ABOUTME: This file documents what changed per phase, why, and observed effects.
ABOUTME: It is used by Claude Code to maintain a research log of technique experiments.

# Changelog

## Phase 0: Clean Neural Baseline
- **Technique:** Dead code removal (n-gram cache, phrase cache, distillation, prefill)
- **What changed:** Removed invalidated eval-time n-gram cache and related code paths
- **Why:** Competition organizers invalidated all n-gram cache submissions due to improper probability normalization. A/B testing confirmed phrase cache actively harmful. Line budget needed for competitive techniques.
- **Expected effect:** No change to neural eval metrics. ~130 lines freed for phases 1-8.
- **Observed effect:** 1307 lines (from 1487). 193 lines freed. Syntax check passes. No remaining references to removed code.
