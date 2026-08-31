# Occlusion Renderer for autonomous driving

KITTI Tracking의 학습 프레임에 시간적으로 연속된 car/person occluder를 합성하고,
YOLOX-X detector와 ByteTrack의 occlusion 강건성을 평가하는 파이프라인입니다. 현재 기본
설정은 [`configs/default.yaml`](configs/default.yaml)이며, 객체 추출 과정에서 GT bbox,
visibility, class를 사용하지 않습니다.

```text
KITTI train + MOT17 FRCNN frames
  -> Co-DETR detections
  -> ByteTrack association (ReID 없음)
  -> detector bbox crop + SAM3 mask
  -> confidence >= 0.60인 최장 연속 구간
  -> random/mid jitter copy-paste
  -> 원본 KITTI + pasted frame 학습
  -> YOLOX-X detection / ByteTrack tracking 평가
```

## 현재 기본 동작

- Co-DETR ViT-L은 프레임별 bbox, confidence, class를 생성합니다. score 0.10 이상을
  ByteTrack에 넘겨 낮은 score의 2차 association도 허용합니다.
- ByteTrack은 위치·IoU·motion만으로 tracklet을 연결하며 appearance/ReID 모델은 쓰지
  않습니다. downstream detector는 프레임 단위로 학습하므로 identity purity는 진단값일
  뿐, tracklet 채택 조건이 아닙니다.
- detector bbox를 crop한 뒤 SAM3가 RGBA mask를 만듭니다. 붙이는 객체의 class는 score와
  무관하게 Co-DETR의 최고-confidence label을 사용하고, label 자체가 없는 예외에만 bbox
  aspect ratio를 fallback으로 사용합니다. bus/truck detection은 pasted `car`로 합칩니다.
- 품질 필터는 각 프레임의 detector confidence가 0.60 이상인 구간 중 첫 번째 최장 연속
  구간을 보존합니다. 30프레임 미만이면 tracklet 전체를 제외합니다. 현재 raw 200개에서
  160개 tracklet, 10,052프레임에서 8,114프레임이 남았습니다.
- 기본 합성은 기존 성능이 검증된 random `mid` jitter이고, 데이터셋 구성은 `append`입니다.
  따라서 treatment는 원본 3,644장보다 많은 5,838장으로 학습됩니다.
- detector head와 평가는 KITTI의 `car`, `truck`, `person`, `bicycle` 4-class를 유지하지만,
  pasted 객체는 데이터가 충분한 `car`, `person`만 사용합니다.
- KITTI/MOT17 GT는 train/eval split, YOLOX 학습 label, 선택 완료 후 audit, 최종 평가에만
  사용됩니다. 자동 crop·class 결정·confidence filtering에는 들어가지 않습니다.

## 측정 결과

동일한 KITTI eval 4,364장에서 COCO-style bbox metric을 측정했습니다. 두 모델 모두
COCO-pretrained YOLOX-X를 60 epoch fine-tuning한 seed 0 EMA checkpoint이며, treatment에는
자동 추출·confidence filtering한 pasted frame을 추가했습니다. real jitter는 적용하지
않고 기본 random `mid` jitter를 사용했습니다.

| 학습 데이터 | 이미지 | 전체 mAP | Occlusion AP | Non-Occlusion AP |
|---|---:|---:|---:|---:|
| 원본 KITTI train | 3,644 | 37.88 | 16.87 | 45.79 |
| **원본 + automatic/confidence-filtered paste** | **5,838** | **41.17 (+3.29)** | **18.18 (+1.31)** | **50.46 (+4.68)** |

추가 지표도 AP50 58.89→64.67, occluded AR100 25.90→29.01, occluded Recall50
41.31→47.89로 상승했습니다. 전체 수치와 class별 결과는 `artifacts/` 아래의 detection
metric `summary.json`에 저장되어 있습니다.

confidence filtering 전후의 동일 seed tracking 결과는 다음과 같습니다. 이 비교에서는
두 모델 모두 `append`, random `mid`, ep60 조건을 사용했습니다.

| automatic pool | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| raw | 55.787 | 52.242 | 60.455 | 60.180 | 71.557 | 200 |
| **confidence-filtered** | **55.953** | 52.203 | **60.894** | **60.450** | **71.717** | **166** |

### Detection metric 정의

- 전체 mAP: 전체 GT에 대한 COCO bbox AP, IoU 0.50:0.95 평균입니다.
- Occlusion AP: KITTI `occluded`가 1(partly) 또는 2(fully/large occlusion)인 GT만 남겨
  같은 COCO AP를 다시 계산합니다.
- Non-Occlusion AP: KITTI `occluded == 0` GT만 남겨 계산합니다.
- `occluded == 3`(unknown) 696개는 전체 mAP에는 포함하지만 두 subset에서는 제외합니다.
- subset GT 수는 occluded 12,353개, non-occluded 14,347개입니다.
- AP50/AP75는 각각 IoU 0.50/0.75의 AP, AR100은 이미지당 최대 100 detection에서
  IoU 0.50:0.95 평균 recall, Recall50은 IoU 0.50 recall입니다.

전체 AP는 표준 COCO 방식이고 occlusion별 AP는 같은 evaluator를 KITTI occlusion subset에
적용한 프로젝트 분석 지표입니다. KITTI 공식 leaderboard metric 그 자체는 아닙니다.
`DontCare`와 sitting-person 처리는 tracking 평가와 동일하게 적용합니다.

## 환경 구성

서로 호환되지 않는 버전 제약 때문에 세 환경을 분리합니다.

| 환경 | 역할 | 정의 |
|---|---|---|
| `kds-occlusion` | 합성, ByteTrack/YOLOX, audit, 평가 | [`environment.yml`](environment.yml), [`requirements.txt`](requirements.txt) |
| `kds-codetr` | legacy MMCV/MMDetection 기반 Co-DETR 추론 | [`environment.codetr.yml`](environment.codetr.yml) |
| `kds-sam3` | 공식 SAM3 추론 | [`environment.sam3.yml`](environment.sam3.yml) |

### 기본 환경 설치

`.condarc`의 `envs_dirs`와 `pkgs_dirs`는 절대경로이므로 새 머신에서는 먼저 현재 저장소
경로에 맞게 수정합니다. CUDA 버전은 서버 driver에 맞게 조정하십시오.

```bash
REPO=/path/to/Occlusion_renderer
BYTETRACK=/path/to/ByteTrack
BOXMOT=/path/to/boxmot

cd "$REPO"
export CONDARC="$REPO/.condarc"
conda env create -f environment.yml
source scripts/activate_kds.sh

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install -v -e "$BYTETRACK" --no-build-isolation
python -m pip install -e "$BOXMOT"
python -m pip install -e . --no-build-isolation --no-deps
```

이후 작업에서는 저장소 경로와 관계없이 다음 스크립트로 기본 환경을 활성화합니다.

```bash
source /path/to/Occlusion_renderer/scripts/activate_kds.sh
```

Co-DETR는 Python 3.8, PyTorch 1.11, MMCV 1.5.0을 사용하는 별도 환경입니다. 설치와
checkpoint 준비는 [Co-DETR 자동 추출 가이드](docs/CODETR_WORKFLOW.md)를 따릅니다.
SAM3 설치 및 smoke test는 [SAM3 가이드](docs/SAM3_WORKFLOW.md)에 정리되어 있습니다.

## 데이터와 외부 경로

[`configs/base.yaml`](configs/base.yaml)의 `paths.*`가 모든 경로의 기준입니다. 저장소 밖의
dataset/repository는 환경변수로 덮어쓸 수 있습니다.

| 환경변수 | 용도 | 기본값 |
|---|---|---|
| `KDS_MOT17` | MOT17 train | `~/tmp_SwapPatch/data/MOT17` |
| `KDS_BYTETRACK_REPO` | ByteTrack/YOLOX checkout | `~/ByteTrack` |
| `KDS_BOXMOT_REPO` | BoxMOT checkout | `~/boxmot` |
| `KDS_TRACKEVAL_REPO` | TrackEval checkout | `~/BankTweak/TrackEval` |
| `KDS_YOLOX_X_CHECKPOINT` | COCO-pretrained YOLOX-X | `$KDS_BYTETRACK_REPO/pretrained/yolox_x.pth` |

저장소 내부 기본 위치는 다음과 같습니다.

```text
datasets/KITTI/training/{image_02,label_02}
third_party/Co-DETR
third_party/sam3
weights/codetr/pytorch_model.pth
weights/sam3/sam3.pt
```

KITTI Tracking은 training image와 label을 모두 준비하고, MOT17은 train의 FRCNN view만
사용합니다. 준비 후 해석된 경로를 먼저 점검합니다.

```bash
python scripts/preflight.py --config configs/default.yaml
```

## 실행 방법

모든 명령은 `kds-occlusion` 환경에서 실행합니다. Co-DETR와 SAM3가 필요한 command는
설정된 전용 Python으로 자동 재실행됩니다.

### Split 고정

자동 추출 전에 `configs/default.yaml`을 사용하는 split 생성 entry point로 고정 split을
한 번 만듭니다. crop을 허용하는 KITTI sequence는 train split의 부분집합이어야 하며 eval과
겹치면 즉시 실패합니다. 이미 생성한 `split.json`은 이후 모든 실행에서 그대로 재사용합니다.

### 자동 detection과 RGBA tracklet 생성

Co-DETR inference는 KITTI 전체가 아니라 고정된 KITTI train/crop-allowed sequence와
MOT17 FRCNN train frame에서 수행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --dataset kitti
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --dataset mot17

# detection JSON -> ByteTrack candidate만 점검
kds-build-detector-tracklets --inventory-only

# candidate bbox -> SAM3 RGBA; 완료 후 confidence filter도 자동 실행
CUDA_VISIBLE_DEVICES=0,3,4,5 kds-build-detector-tracklets --workers 4
```

multi-GPU SAM3 shard는 seed로 정한 원래 candidate 순서대로 합쳐지므로 scheduling 순서가
최종 선택을 바꾸지 않습니다. raw pool은 보존되고 filtered pool은 별도 디렉터리에 생성됩니다.

필터만 다시 실행하거나 GT를 selection에 되먹이지 않는 post-hoc audit 및 동일 장면 비교
영상을 만들려면 다음을 사용합니다.

```bash
kds-filter-detector-tracklets
kds-compare-tracklet-pools --config configs/default.yaml
kds-audit-confidence-filter --config configs/default.yaml
kds-visualize-confidence-filter --config configs/default.yaml
```

비교 영상은 같은 raw tracklet, background frame, 위치, 크기를 유지합니다. 탈락 tracklet도
filter 전 panel에는 그대로 표시되므로 전후 차이가 confidence decision 하나로 제한됩니다.

### Occlusion 합성과 학습 데이터 구성

```bash
kds-synth-events
kds-build-ab
```

기본 treatment는 원본 train과 pasted frame을 합치는 `append`입니다. `replace` 비교가
필요한 경우에만 `kds-build-ab --paste-mode replace`를 명시합니다. 합성 label은 원본 victim의
amodal bbox를 유지하고 synthetic occluder GT를 추가합니다. 실제 alpha mask가 victim bbox를
덮는 비율 `rho`로 사건을 검증하며, paste는 large-first hard paste입니다.

### YOLOX-X 학습

```bash
# global batch size 4; 네 GPU에서 GPU당 batch 1
CUDA_VISIBLE_DEVICES=0,3,4,5 kds-train-yolox \
  --condition baseline --aug full --seed 0

CUDA_VISIBLE_DEVICES=0,3,4,5 kds-train-yolox \
  --condition treatment --paste-mode append --tag conf60 --aug full --seed 0
```

GPU 수가 달라도 global batch size와 learning-rate policy는 한 GPU 학습과 동일하게 유지됩니다.
중단된 동일 실험은 `--resume`으로 `latest_ckpt.pth.tar`에서 이어갈 수 있습니다.

### Tracking과 detection 평가

```bash
CUDA_VISIBLE_DEVICES=0,3,4,5 kds-eval-boxmot \
  --condition treatment --paste-mode append --tag conf60 \
  --epoch ep60 --seed 0 --workers 4

CUDA_VISIBLE_DEVICES=0,3,4,5 kds-eval-detection-metrics --workers 4
```

tracking은 YOLOX detection에 ReID를 끈 BoxMOT ByteTrack을 적용하고 TrackEval로 HOTA, CLEAR,
Identity metric을 계산합니다. detection command는 baseline과 confidence-filtered treatment를
같은 eval JSON에서 추론한 뒤 전체/occluded/non-occluded COCO bbox metric을 함께 저장합니다.

## Jitter 설정

기본 `random/mid`는 pasted crop마다 color, brightness, contrast, sharpness, horizontal flip,
scale, rotation을 추첨합니다. 실제 환경을 모사하는 옵션은 한 사건 안에서 같은 scenario를
유지합니다.

| scenario | 구현 |
|---|---|
| day | 약한 brightness 변화 |
| night | 강한 brightness 감소 |
| tunnel | 밝기 감소, warm tone, vignette |
| rain | RGB에만 sparse 2×2 local displacement |
| snow | 약한 2×2 displacement와 sparse white RGB pixel |

rain/snow도 alpha, 객체 geometry, placement용 `rho`는 바꾸지 않습니다. 기본 jitter와 다섯
scenario를 같은 tracklet·배경·위치·크기로 비교하는 영상은 아래 command로 생성합니다.

```bash
kds-visualize-real-jitter
```

실제 합성에 적용하려면 `configs/`의 real-jitter 파생 config처럼
`tracklet_synthesis.paste_jitter.mode: real`을 선택합니다. 이 옵션의 구현과 시각화는
완료되었지만, 위 결과표의 학습에는 사용하지 않았습니다.

## GPU 예산

[`configs/base.yaml`](configs/base.yaml)의 `resources`가 GPU 예산을 검사합니다.

- 계정 전체 동시 사용 상한: 6 GPU
- 이 프로젝트의 SAM3, train, eval 한 command 상한: 4 GPU
- Co-DETR detection: 1 GPU
- `CUDA_VISIBLE_DEVICES`의 개수와 `--workers`는 일치해야 합니다.
- 다른 작업이 이미 사용 중인 GPU도 계정 상한에 포함됩니다. 예시의 `0,3,4,5`는 1,2번이
  다른 작업에 사용 중인 경우이며, 실제 유휴 GPU에 맞게 바꿔야 합니다.

상한을 넘기는 실행은 시작 전에 실패하므로 optimizer state나 shard가 일부만 만들어지는
상황을 방지합니다.

## 설정과 주요 산출물

| 목적 | 설정 |
|---|---|
| 현재 기본 자동 파이프라인 | [`configs/default.yaml`](configs/default.yaml) |

`configs/`의 파생 YAML은 각각 filtering 없는 raw 자동 추출, confidence filter, real jitter,
occlusion별 detection 평가를 재현합니다. 기본 실행은 별도 `--config` 없이 위 default 설정을
사용합니다.

주요 output은 config에 선언된 `artifacts/` 하위에 기록됩니다.

| 산출물 | 내용 |
|---|---|
| `codetr_detections/` | 환경 중립 Co-DETR frame JSON |
| `codetr_bytetrack_candidates/` | ReID 없는 ByteTrack candidate |
| `codetr_sam3_tracklets/` | filtering 전 RGBA tracklet |
| `codetr_sam3_tracklets_conf60/` | 채택/탈락 decision과 filtered tracklet |
| `confidence_filter_videos/` | 동일 장면 filter 전후 영상과 contact sheet |
| `tracklet_synthetic_conf60/` | 합성 frame, annotation, event, QC |
| `yolox_data_conf60/` | baseline/treatment/eval COCO JSON |
| `yolox_runs_conf60/` | YOLOX checkpoint와 log |
| `tracking_conf60/` | ByteTrack prediction과 TrackEval 결과 |
| `detection_metrics_random_mid/` | prediction JSON과 전체/occlusion별 metric |

## 구현 위치

- 자동 source enumeration: [`data/detection_sources.py`](data/detection_sources.py)
- Co-DETR export: [`mining/codetr.py`](mining/codetr.py)
- ByteTrack association: [`mining/detected_tracklets.py`](mining/detected_tracklets.py)
- SAM3 pool과 자동 filter 연결: [`pool/build_detector_tracklet_pool.py`](pool/build_detector_tracklet_pool.py)
- confidence filter: [`pool/filter_detector_tracklet_pool.py`](pool/filter_detector_tracklet_pool.py)
- filter audit/비교: [`eval/audit_confidence_filter.py`](eval/audit_confidence_filter.py),
  [`eval/compare_tracklet_pools.py`](eval/compare_tracklet_pools.py)
- random/real jitter: [`synth/paste_jitter.py`](synth/paste_jitter.py)
- event 단위 scenario와 합성: [`synth/event_pipeline.py`](synth/event_pipeline.py)
- multi-GPU 예산 검사: [`common/gpu_budget.py`](common/gpu_budget.py)
- occlusion별 detection metric: [`eval/run_detection_metrics.py`](eval/run_detection_metrics.py)