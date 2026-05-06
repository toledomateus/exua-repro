#!/usr/bin/env bash
# copy_checkpoints.sh
#
# Reads all checkpoint paths from r1_results.csv and rq2_results.csv,
# plus the SASRec baselines, and copies every .pth file from saved/ to saveds/
# preserving the directory structure.
#
# Usage (run from the root of the repo):
#   bash scripts/copy_checkpoints.sh

set -euo pipefail

SRC_ROOT="."          # repo root where saved/ lives
DST_ROOT="saveds"     # destination folder

MISSING=0
COPIED=0
SKIPPED=0

copy_path() {
    local path="$1"
    local src="${SRC_ROOT}/${path}"
    local dst="${DST_ROOT}/${path}"

    if [[ ! -f "$src" ]]; then
        echo "  [MISSING]  $src"
        ((MISSING++)) || true
        return
    fi

    if [[ -f "$dst" ]]; then
        echo "  [SKIP]     $dst already exists"
        ((SKIPPED++)) || true
        return
    fi

    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "  [COPIED]   $dst"
    ((COPIED++)) || true
}

echo "=== Collecting paths from CSVs ==="

# Extract checkpoint_path column (index 6, 0-based) from both CSVs,
# skip header, split space-separated DE ensemble paths, deduplicate.
ALL_PATHS=$(python3 - <<'PYEOF'
import csv, sys

files = ["results/r1_results.csv", "results/rq2_results.csv"]
paths = set()

for fname in files:
    try:
        with open(fname) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for p in row["checkpoint_path"].strip().split():
                    if p.endswith(".pth"):
                        paths.add(p)
    except FileNotFoundError:
        print(f"WARNING: {fname} not found, skipping.", file=sys.stderr)

# SASRec baselines (not always in the CSVs)
baselines = [
    "saved/SASRec-Feb-18-2026_10-53-03.pth",
    "saved/SASRec-Feb-17-2026_12-15-15.pth",
    "saved/SASRec-Mar-22-2026_13-57-46.pth",
    "saved/SASRec-Mar-23-2026_14-28-26.pth",
    "saved/SASRec-Mar-24-2026_07-23-07.pth",
    "saved/SASRec-Mar-19-2026_09-59-32.pth",
    "saved/SASRec-Mar-18-2026_11-55-28.pth",
    "saved/SASRec-Mar-18-2026_11-55-54.pth",
    "saved/SASRec-Mar-18-2026_11-56-10.pth",
    "saved/SASRec-Mar-18-2026_11-56-30.pth",
    "saved/SASRec-Mar-11-2026_13-03-57.pth",
    "saved/SASRec-Mar-11-2026_13-09-16.pth",
    "saved/SASRec-Mar-11-2026_13-09-48.pth",
    "saved/SASRec-Mar-11-2026_13-10-06.pth",
    "saved/SASRec-Mar-11-2026_13-10-16.pth",
    "saved/SASRec-Mar-12-2026_10-56-38.pth",
    "saved/SASRec-Mar-13-2026_10-25-01.pth",
    "saved/SASRec-Mar-13-2026_10-27-28.pth",
    "saved/SASRec-Mar-13-2026_10-27-56.pth",
    "saved/SASRec-Mar-13-2026_10-28-19.pth",
    "saved/SASRec-Mar-26-2026_20-47-33.pth",
    "saved/SASRec-Mar-11-2026_13-29-59.pth",
    "saved/SASRec-Mar-11-2026_13-30-10.pth",
    "saved/SASRec-Mar-11-2026_13-30-17.pth",
    "saved/SASRec-Mar-11-2026_13-30-24.pth",
    "saved/SASRec-Mar-25-2026_15-17-11.pth",
    "saved/SASRec-Mar-25-2026_15-17-56.pth",
    "saved/SASRec-Mar-25-2026_15-18-09.pth",
    "saved/SASRec-Mar-25-2026_15-19-28.pth",
    "saved/SASRec-Mar-25-2026_15-19-37.pth",
]
paths.update(baselines)

for p in sorted(paths):
    print(p)
PYEOF
)

echo ""
echo "=== Copying files ==="

while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    copy_path "$path"
done <<< "$ALL_PATHS"

echo ""
echo "=== Done ==="
echo "  Copied:  $COPIED"
echo "  Skipped: $SKIPPED (already existed)"
echo "  Missing: $MISSING (not found in saved/)"