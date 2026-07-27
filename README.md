# Occlusion Renderer for autonomous driving
SAM3로 분할한 동일 identity의 연속 RGBA tracklet(MOT17 human · KITTI car/human)을 KITTI
Tracking 학습 시퀀스에 시간적으로 끊김 없이 copy-paste하여 **연속적인 occlusion**을
만들고, 자동 생성한 확장 GT로 COCO-pretrained YOLOX-X detector를 fine-tuning한 뒤 동일
ByteTrack(from BoxMOT)으로 원본 KITTI eval 시퀀스에서 tracking 강건성을 검증합니다.

추후 tracker, detector, dataset, render 세부 사항 변경 등 성능 개선의 여지가 충분합니다.

---

## Results (KITTI half train/half valid based on MOT protocol)

| Detector | HOTA | DetA | AssA | MOTA | IDF1 | 
|---|---:|---:|---:|---:|---:|
| Baseline fine tuned | 54.42 | 49.12 | 60.94 | 56.07 | 70.24 | 
| **Fine tuned by our method** | **55.69** | **51.09** | **61.50** | **57.82** | **71.59** | 

Bytetrack에서 언급한 MOT 내장 detector 학습 관례를 따르되, baseline은 단순 color/flip jitter를 통한 augmentation, out method는 occlusion renderer를 통한 augmentation. 두 방법의 train dataset 총 크기는 동일하다.

---

## 1. Detail

- 개별 crop을 매 프레임 교체하지 않고 **동일 occluder identity의 tracklet**을 연속 paste.
- tracklet 길이는 **target 10FPS 기준 30~100프레임 가변**(min 3초). MOT17은 source FPS(14/25/30)를
  10FPS로 resampling. visibility>=0.8은 샘플 프레임에서 검사(인접 프레임 대체 허용).
- SAM3 실패 시 tracklet 전체 폐기가 아니라 **가장 긴 통과 sub-run(>=30프레임)** 을 보존.
- occluder 클래스 = **car/person 2-class (실측 65/35 재정규화)**. truck·bicycle은 KITTI train에
  >=30프레임 clean 연속 run이 극히 적어(truck 0개, bicycle 1개) occluder pool에서 제외합니다.
  단 detector head는 4-class(car/truck/person/bicycle)를 유지합니다(eval GT에 존재). 이 구성은
  KITTI에 맞춰진 값이라 추후 변동 가능성 있습니다.
- 크기 = **victim 상대 스케일**: `occluder높이 = victim높이 × (canonical[occluder]/canonical[victim])`.
- 배치는 peak 프레임에서 occluder를 **victim의 중앙(bottom-center)에 정렬**하고, 전후 프레임은
  occluder 자신의 궤적으로 지나가게 하여 가림이 들어왔다 빠지도록 합니다. **여러 scale/시점을
  탐색해 목표 peak rho**를 맞춘 완결된 사건(0→peak→0, rho>=0.1이 8–20프레임, 단일 peak,
  peak<=0.80)만 채택합니다. rho는 bbox-proxy(victim bbox 대비).
- large-first paste, hard paste(`blend_method: none`), 동시 occluder<=2.
- victim detector bbox 정책 = **amodal_original**(원본 bbox 유지, visible_bbox는 별도 저장).


---

## 2. Environments

두 환경을 분리합니다.
- `kds-occlusion` — ByteTrack/YOLOX, 데이터 변환, 합성, 라벨링, 평가
- `kds-sam3` — 공식 SAM3 추론 (Python 3.12, NumPy 1.26 계열)

```bash
cd /path/to/Occlusion_renderer
export CONDARC=$PWD/.condarc
conda env create -f environment.yml           # kds-occlusion
source scripts/activate_kds.sh

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118      # 서버 드라이버에 맞게 조정
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install -v -e /path/to/ByteTrack --no-build-isolation
python -m pip install -e /path/to/boxmot
python -m pip install -e . --no-build-isolation --no-deps
```

SAM3 환경(`environment.sam3.yml`) 설치와 smoke 절차는 [docs/SAM3_WORKFLOW.md](docs/SAM3_WORKFLOW.md)에 기록되어 있습니다.

---

## 3. Dataset · dependency · checkpoint download

경로는 `configs/base.yaml`의 `paths.*`에서 지정합니다.

### 3.1 KITTI Tracking (배경/GT/eval)
[KITTI Tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)에서 Download left color images of tracking data set, Download training labels of tracking data set을 받아 아래 구조로 풉니다.

```bash
unzip -n datasets/KITTI/data_tracking_image_2.zip -d datasets/KITTI
unzip -n datasets/KITTI/data_tracking_label_2.zip -d datasets/KITTI
# datasets/KITTI/training/{image_02/0000..0020, label_02/0000.txt..0020.txt}
```

### 3.2 MOT17 (human tracklet source)
[MOTChallenge MOT17](https://www.codabench.org/competitions/10049/#/pages-tab) train을 받아 풀고
`configs/base.yaml`의 `paths.mot17`을 그 경로로 설정합니다(`train/MOT17-XX-FRCNN/{img1,gt,seqinfo.ini}`).
FRCNN detector view만 사용합니다(DPM/SDP 중복 제거).

### 3.3 SAM3 checkpoint
Hugging Face 승인 후 내려받습니다.
```bash
mkdir -p weights/sam3
.conda-envs/kds-sam3/bin/hf auth login
.conda-envs/kds-sam3/bin/hf download facebook/sam3 sam3.pt config.json --local-dir weights/sam3
```

### 3.4 YOLOX-X · ByteTrack · BoxMOT · TrackEval
```bash
# ByteTrack (YOLOX 학습/추론 뼈대) — 위 2절에서 editable 설치
git clone https://github.com/ifzhang/ByteTrack   # paths.bytetrack_repo
# COCO-pretrained YOLOX-X 가중치(yolox_x.pth)는 YOLOX 공식 model zoo
# (Megvii-BaseDetection/YOLOX) releases에서 받아 ByteTrack/pretrained/yolox_x.pth에 둡니다.
# (paths.coco_pretrained_yolox_x)

# BoxMOT (ByteTrack 추적기 구현)
git clone https://github.com/mikel-brostrom/boxmot  # paths.boxmot_repo

# TrackEval (HOTA/CLEAR/Identity)
git clone https://github.com/JonathonLuiten/TrackEval  # paths.trackeval_repo
```

`configs/base.yaml`의 `paths.{bytetrack_repo, boxmot_repo, trackeval_repo, coco_pretrained_yolox_x,
sam3_checkpoint, kitti_tracking, mot17}`를 실제 경로로 맞춥니다.

---

## 4. Workflow

### 4.1 고정 split (train-only crop / eval 누수 차단)
```bash
kds-phase1-split --config configs/phase1_kitti.yaml   # artifacts/phase1/split.json
```
`crop_allowed_sequences`는 train의 부분집합이어야 하며 eval과 겹치면 즉시 실패합니다.

### 4.2 SAM3 tracklet pool 생성 (GPU 필요)
```bash
# 후보 검증(SAM3 미로드)
kds-build-tracklets --config configs/phase1_kitti.yaml --inventory-only         # MOT17
kds-build-kitti-sam3-pool --config configs/phase1_kitti.yaml --tracklet --inventory-only  # KITTI

# 실제 RGBA pool (SAM3 로드; kds-sam3 환경으로 자동 재실행)
CUDA_VISIBLE_DEVICES=0 kds-build-tracklets --config configs/phase1_kitti.yaml               # MOT17 person
CUDA_VISIBLE_DEVICES=1 kds-build-kitti-sam3-pool --config configs/phase1_kitti.yaml --tracklet  # KITTI car/person
```
출력: `artifacts/phase1/mot17_sam3_tracklets/tracklets.json`,
`artifacts/phase1/kitti_sam3_tracklets/tracklets.json` (+ tracklet별 RGBA PNG).

### 4.3 이벤트 기반 합성 (occlusion paste)
```bash
kds-synth-events --config configs/phase1_kitti.yaml --max-sequences 1   # 1 시퀀스 smoke
kds-synth-events --config configs/phase1_kitti.yaml                     # 전체 train split
```
출력 `artifacts/phase1/tracklet_synthetic/`: `annotations.json`(real victim 확장 GT + synthetic
occluder GT), `occluder_tracks.json`, `events.json`, `scheduling.jsonl`, `qc.json`, `frames/`.

### 4.4 equal-budget A/B 데이터셋
```bash
kds-build-ab --config configs/phase1_kitti.yaml
```
- **Treatment** = 원본 train + **paste 프레임만**(`dataset.synthetic_frames: paste_only`)
- **Baseline** = 원본 train + paste 수만큼의 flip/color-jitter 프레임(`dataset.baseline_equal_budget: true`)
- 산출 JSON: `artifacts/phase1/yolox_data/annotations/{baseline_train,treatment_train,eval,kitti_train}.json`

### 4.5 detector fine-tuning (YOLOX-X, COCO에서)
```bash
# condition ∈ {kitti(원본만), baseline(+jitter frames), treatment(+paste)}
# aug ∈ {none, jitter(flip+HSV), full(+mosaic/mixup, 마지막 10ep off)}
CUDA_VISIBLE_DEVICES=0 kds-train-yolox --condition treatment --aug full --seed 0
CUDA_VISIBLE_DEVICES=1 kds-train-yolox --condition baseline  --aug full --seed 0
# smoke: --max-epoch 1
```
출력 `artifacts/phase1/yolox_runs/phase1_<label>_<aug>_seed<seed>/`:
`latest_ckpt.pth.tar`(=ep60, EMA), `last_mosaic_epoch_ckpt.pth.tar`(=ep50 경계 스냅샷).

### 4.6 tracking evaluation (BoxMOT ByteTrack + TrackEval)
```bash
# eval 9시퀀스 원본 KITTI에서 YOLOX 추론 → ByteTrack → 클래스별 TrackEval(HOTA/CLEAR/Identity)
CUDA_VISIBLE_DEVICES=0 kds-eval-boxmot --condition treatment --aug full --epoch ep60 --seed 0
# 참조: COCO-pretrained
kds-eval-boxmot --model-source coco-pretrained
```
ByteTrack 파라미터는 `configs/*.yaml`의 `tracker.*`(frame_rate10 · min_conf0.10 · track_thresh0.45 ·
match_thresh0.80 · track_buffer30 · per_class · reid off). 결과는
`artifacts/phase1/tracking/<run_name>/trackers/KDS_<CLASS>-eval/<run_name>/pedestrian_{summary.txt,detailed.csv}`.

### 4.7 비교표 집계
```bash
kds-aggregate --runs phase1_coco_pretrained \
  phase1_baseline_full_seed0_ep60 phase1_treatment_full_seed0_ep60 \
  --out artifacts/phase1/results.md
```
per-class TrackEval 출력에서 **전체(combined)** 를 detection-weighted(alpha별 TP/FN/FP + TP-weighted
AssA)로 조합하고 MOTA/IDF1은 count 재계산합니다. `fine_tuned_baseline.md`(COCO
및 기존 KITTI fine-tune 기준)와 동일 산출식이라 직접 비교 가능합니다.

---

## 5. 주요 config 노브 (`configs/phase1_kitti.yaml`)

| 섹션 | 키 | 의미 |
|---|---|---|
| `tracklet_pool` | `target_fps, min_frames, max_frames, visibility_substitution_window` | MOT17 FPS-aware 가변길이 pool |
| `kitti_sam3_pool` | `target_fps, min_frames, max_frames, prompts` | KITTI multi-class tracklet pool |
| `tracklet_synthesis` | `class_ratio, class_height_factor, scale_search_multipliers, peak_rho_distribution, peak_rho_max, effective_event_frames, victim_events_per_100_frames, victim_detector_bbox_policy` | 스케줄러/배치/rho 정책 |
| `dataset` | `synthetic_frames(paste_only|all), baseline_equal_budget` | A/B 조립 |
| `train` | `epochs, batch_size, input_size, fp16, output_dir` | YOLOX-X 학습 |
| `tracker` / `detector` | ByteTrack · 추론 threshold | 평가 |

occlusion_level band(0.20/0.35/0.65)와 peak rho band(mild/moderate/heavy)는 정렬되어 있습니다.

---

## 6. Structure

```text
data/mot17.py               # MOT17 FPS-aware 가변길이 tracklet 후보
data/kitti_tracking.py      # KITTI Tracking → temporal COCO
pool/build_tracklet_pool.py # MOT17 SAM3 RGBA pool (변수 길이 + 최장 sub-run)
pool/build_kitti_sam3_pool.py # KITTI multi-class SAM3 tracklet pool
segment/sam3.py             # 공식 SAM3 text-prompt adapter
synth/scheduler.py          # 이벤트 계획(밀도/클래스/재사용/동시성)
synth/placement.py          # target-rho + victim-relative scale + acceptance
synth/geometry.py           # occluder box 매핑 + placement 탐색
synth/event_pipeline.py     # 통합 합성 렌더 + 확장 GT + QC
label/{compute,events}.py   # frame occlusion·bbox 정책·temporal event
data/build_phase1.py        # equal-budget baseline/treatment 조립
train/{run.py, yolox_x_kitti.py} # YOLOX-X fine-tune + aug 토글
eval/run_boxmot.py          # ByteTrack 추론 + TrackEval layout
eval/aggregate.py           # per-class → 전체 비교표
```
