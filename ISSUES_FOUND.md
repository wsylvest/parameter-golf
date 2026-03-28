ABOUTME: This file logs issues discovered during phase execution but not fixed in that phase.
ABOUTME: It is used by Claude Code to track technical debt and deferred work.

# Issues Found

## From A/B Testing (pre-Phase 0)
- Order 2 (bigrams) hurts by -1.664 bpb due to concentration=50 overwhelming neural prior
- All n-gram cache "improvements" were artifacts of hash collision density, not prediction quality
- ADAM_WD env var exists but WD should only apply to Muon matrix params
