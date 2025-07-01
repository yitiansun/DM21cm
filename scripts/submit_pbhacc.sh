#!/usr/bin/env bash
# submit_all.sh — dispatch submit_inj.sh for every injection variant

set -euo pipefail

variants=(PRc23 PRc14 PRc29 PRc23dm PRc23dp PRc23H PRc23B BHLl2)

for v in "${variants[@]}"; do
    tmp=$(mktemp)
    trap 'rm -f "$tmp"' RETURN      # always clean up temp file

    # Substitute every literal occurrence of "PRc23"
    sed "s/MODEL/${v}/g" submit_inj.sh > "$tmp"

    sbatch "$tmp"                   # submit to Slurm
    echo " → submitted variant: $v"
done
