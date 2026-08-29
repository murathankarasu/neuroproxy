#!/usr/bin/env bash
# Fetch the pretrained rPPG checkpoints used by the neural methods.
#
# These are NOT redistributed with this repository. They come from
# rPPG-Toolbox under the Responsible AI Source Code License (RAIL), whose
# behavioural restrictions must be passed through to downstream users --
# see NOTICE.md. Downloading them here means accepting those terms.
#
# Measured caveat before you rely on them: the pretrained models do not
# transfer to real consumer-webcam recordings. See docs/limitations.md 19.
set -euo pipefail
BASE="https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release"
mkdir -p models
for w in PURE_EfficientPhys.pth UBFC-rPPG_EfficientPhys.pth SCAMPS_EfficientPhys.pth PURE_TSCAN.pth; do
  if [ -s "models/$w" ]; then echo "have  $w"; continue; fi
  echo "fetch $w"
  curl -sL --fail -o "models/$w" "$BASE/$w"
done
echo "done. $(du -sh models | cut -f1) in models/"
