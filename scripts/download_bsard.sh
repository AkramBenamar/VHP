#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download maastrichtlawtech/bsard --repo-type dataset \
      --local-dir data/bsard
else
  git clone https://github.com/maastrichtlawtech/bsard data/bsard
fi
echo "Then check the columns:"
echo "  python -m bsard_vhp.run inspect --data-dir data/bsard"
