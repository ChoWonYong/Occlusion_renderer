# Co-DETR GT-free tracklet workflow

The published `zongzhuofan/co-detr-vit-large-coco` checkpoint uses
MMDetection 2.25.3 and MMCV 1.5.0. It runs in `kds-codetr`, separate from the
current ByteTrack and SAM3 environments. A versioned JSON file is the only
interface between these environments.

## One-time setup

```bash
export CONDARC="$PWD/.condarc"
conda env create -f environment.codetr.yml
conda run -p "$PWD/.conda-envs/kds-codetr" python -m pip install \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
conda run -p "$PWD/.conda-envs/kds-codetr" python -m pip install \
  mmcv-full==1.5.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
git clone https://github.com/Sense-X/Co-DETR third_party/Co-DETR
conda run -p "$PWD/.conda-envs/kds-codetr" python -m pip install \
  -r third_party/Co-DETR/requirements.txt
mkdir -p weights/codetr
huggingface-cli download zongzhuofan/co-detr-vit-large-coco pytorch_model.pth \
  --local-dir weights/codetr
```

The official repository bundles its MMDetection fork. Do not install modern
MMDetection into this environment.

## Phase 1 execution

Co-DETR inference uses one GPU. Independent SAM3 candidate shards may use up to
four GPUs. Before launching either stage, count the distinct GPUs already used
by this account. The account-wide total, including unrelated jobs, must remain
at most six. If unrelated jobs occupy GPU 1 and GPU 2, the four-GPU example
below may use GPU 0, 3, 4, and 5.

```bash
# Co-DETR inference -> environment-neutral JSON
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/phase1_detector.yaml --dataset kitti
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/phase1_detector.yaml --dataset mot17

# ByteTrack candidates, no SAM3 load and no GT selection
kds-build-detector-tracklets --config configs/phase1_detector.yaml --inventory-only

# Detector bbox crop -> SAM3 -> RGBA tracklets. Shards are merged back in the
# original seeded candidate order, so GPU scheduling does not change selection.
CUDA_VISIBLE_DEVICES=0,3,4,5 kds-build-detector-tracklets \
  --config configs/phase1_detector.yaml --workers 4

# Post-hoc audit only; GT never feeds the extraction path
kds-compare-tracklet-pools --config configs/phase1_detector.yaml
```

The primary Phase-1 audit is `clean_frame_precision`: a selected frame must
match a MOT17 object with visibility at least 0.8, or a KITTI object with
`occluded=0` and truncation at most 0.2. Identity purity is reported only as a
diagnostic because the downstream detector is trained independently per frame.
Phase 2 will use detector/SAM signals to approximate this clean-frame target
without GT and will keep the longest consecutive passing run.

## Phase 2 confidence filtering

Phase 1's raw pool remains immutable. Phase 2 writes a separate pool and a
decision manifest containing every accepted and rejected raw tracklet. The
selection path reads only per-frame Co-DETR confidence: frames below 0.60 are
removed, the first longest consecutive passing run is kept, and runs shorter
than 30 frames are rejected. GT and identity purity are not selection inputs.

```bash
kds-filter-detector-tracklets --config configs/phase2_detector_conf.yaml

# Post-hoc diagnostic only. This reads source GT after selection and cannot
# change the filtered pool.
kds-audit-confidence-filter --config configs/phase2_detector_conf.yaml

# Controlled videos: source detection/mask, raw paste, and filtered paste use
# the same raw tracklet, background frames, position, and scale. Rejected
# tracklets remain visible in the raw-paste panel.
kds-visualize-confidence-filter --config configs/phase2_detector_conf.yaml
```

The video summary records that extra paste jitter is disabled for this
diagnostic, so the only before/after variable is the confidence decision.

Class assignment is independent of the quality threshold. Each Co-DETR query's
highest-confidence detector label is retained even below 0.60 so ByteTrack can
use low-score detections for temporal bridging. Confidence below 0.60 removes a
frame from the final paste pool; it never changes car to person or vice versa.
The aspect heuristic is used only if a detector adapter provides no label.

## Default protocol

`configs/default.yaml` is the command-line default for detector export,
detector-tracklet construction, event synthesis, dataset construction, training,
and evaluation. It selects the GT-free Co-DETR -> ByteTrack -> SAM3 path and the
confidence-filtered (`score >= 0.60`, longest consecutive run >= 30 frames) pool.
After the raw SAM3 pool finishes, `kds-build-detector-tracklets` applies this
configured quality filter automatically. The pasted-object class always comes
from Co-DETR's highest-confidence label; confidence is a quality signal only.

The established random `mid` paste jitter remains the default:

```bash
kds-synth-events
```

Explicit historical phase configs remain available for reproducing the raw and
GT-based arms.

## Phase 3 real jitter

`configs/phase3_real_jitter.yaml` switches only the jitter policy to `real`.
One condition is sampled per pasted event and remains fixed across that event:

- day: brightness adjustment;
- night: stronger brightness reduction;
- tunnel: reduced brightness with warm centre-weighted lighting;
- rain: sparse local 2x2 RGB displacement;
- snow: weaker 2x2 RGB displacement plus sparse white object pixels.

Rain and snow modify RGB only. Alpha, pasted-object geometry, and the rho used by
the placement gate are unchanged. Generate controlled same-scene comparisons
with:

```bash
kds-visualize-real-jitter --config configs/phase3_real_jitter.yaml
```

The videos and contact sheet are written under
`artifacts/phase3/real_jitter_videos/`.

For a minimal smoke test, add `--max-sequences 1 --max-frames-per-sequence 40`
to each Co-DETR command and use a one-GPU SAM3 command with
`--max-tracklets 1` and no `--workers` argument.
