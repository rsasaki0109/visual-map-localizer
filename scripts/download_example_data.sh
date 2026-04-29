#!/usr/bin/env bash
# Fetch a tiny image set for smoke-testing the pipeline.
# (Requires internet. Replace the URL/path with your own dataset as needed.)
set -euo pipefail

DEST="${1:-examples/data/sacre_coeur}"
mkdir -p "$DEST"

cat <<'EOM'
This script is a placeholder.

Recommended toy datasets:
  * Sacre Coeur (hloc demo):
      https://github.com/cvg/Hierarchical-Localization/tree/master/datasets/sacre_coeur
  * Aachen Day-Night v1.1:
      https://www.visuallocalization.net/datasets/

Drop your db images under "$DEST/db_images" and a query under "$DEST/queries".
EOM
