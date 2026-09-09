#!/usr/bin/env bash
# Segment the official-split ByteTrack candidates with SAM3 at 0%, 10% and 20%
# context padding, then score every retained tracklet against KITTI MOTS.
#
# One SAM3 pass per padding produces both crop policies at once:
#   variant_b_context_preserved -> the mask kept over the whole padded crop
#   variant_a_bbox_clipped      -> the same mask re-cropped to the detector bbox
# so the four reported rows come from three runs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.conda-envs/kds-sam3/bin/python"
devices=(0,1 2,3)

run_batch() {
  local percentages=("$@")
  local pids=() status=0 index percentage output_dir
  for index in "${!percentages[@]}"; do
    percentage="${percentages[$index]}"
    output_dir="$repo_root/artifacts/official_split/sam3_context_ablation_${percentage}pct"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES="${devices[$index]}" "$python_bin" \
      -m eval.compare_sam3_crop_policies \
      --config "$repo_root/configs/sam3_context_ablation_${percentage}pct.yaml" \
      --workers 2 >"$output_dir/run.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  return "$status"
}

# 20% is the production padding and gets its own config; each run resumes from
# valid shard JSONL files if interrupted.
run_batch 00 10
CUDA_VISIBLE_DEVICES=0,1 "$python_bin" -m eval.compare_sam3_crop_policies \
  --config "$repo_root/configs/sam3_context_ablation.yaml" --workers 2 \
  >"$repo_root/artifacts/official_split/sam3_context_ablation_20pct/run.log" 2>&1

"$repo_root/.conda-envs/kds-occlusion/bin/python" -m eval.evaluate_selected_tracklet_mots \
  --root "$repo_root" \
  --output "$repo_root/artifacts/official_split/sam3_padding_selected_200_conf60_mots"
