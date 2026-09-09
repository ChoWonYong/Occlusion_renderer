# Occlusion Renderer

## Making DEMO

[DEMO source](https://drive.google.com/file/d/1GOwwfLCQGObRzSRQRyBx_3shXgajgx9y/view?usp=drive_link)
**KDS 대회 당일 호텔 수성 앞에서 휴대폰으로 직접 찍은 non labeled 영상입니다.**

아래 환경·데이터 준비 절차에 따라 `kds-sam3`, SAM3 가중치, KITTI Tracking 이미지와
라벨을 준비한 뒤 저장소 루트에서 실행합니다.
```bash
CUDA_VISIBLE_DEVICES=0 .conda-envs/kds-sam3/bin/python making_demo/make_demo.py
```

make_demo 파일 실행 시, DEMO source 영상 다운로드 → SAM3 `person` 프롬프트로 10fps 사람 tracklet 추출 → KITTI 합성을
한 번에 실행합니다. `making_demo/visualization_tmp/`에 합성(`*_pasted.mp4`)과 대응
원본(`*_original.mp4`) **1쌍**을 저장합니다.
각 영상은 객체 하나만 paste하며 tracklet 시작~끝 구간만 담습니다.
GPU는 SAM3 추출에 필요합니다. 다운로드 영상과 `making_demo/work/`의 추출 결과는
재실행 시 재사용하며, 삭제하면 다시 생성합니다.

---

KITTI Tracking 학습 프레임에 **시간적으로 연속된 car/person occluder를 합성**해
detector와 tracker의 가림(occlusion) 강건성을 높이는 파이프라인입니다.

핵심은 **occlusion augmentation GT 없이 만든다**는 점입니다. 붙일 객체를 고르고, 잘라내고,
클래스를 정하고, 품질을 거르는 전 과정에서 KITTI/MOT17의 bbox·visibility·class와 같은 GT 정보를
전혀 사용하지 않습니다. 기존의 train/test set에다가 임의의 non labeled dataset을 활용할 수 있습니다.

```text
KITTI train(split A) + MOT17 frames
  -> Co-DETR ViT-L detection (score >= 0.10)
  -> ByteTrack association (by motion information) -> tracklet 후보
  -> detector bbox를 20% 확장한 crop으로 SAM3 분할 -> detector bbox로 재-crop
  -> detector confidence >= 0.60인 최장 연속 구간만 채택 (30프레임 미만이면 폐기)
  -> 사건 단위 copy-paste 합성 (random/mid jitter)
  -> 원본 KITTI + pasted frame으로 YOLOX-X 학습
  -> COCO bbox metric / BoxMOT ByteTrack (호환성) + TrackEval 평가
```

설계상의 두 가지 선택:

- **가림 강도 `rho`는 bbox가 아니라 실제 alpha mask로 측정합니다.** bbox proxy는 occluder
  클래스에 따라 0.40~0.72배로 과대평가되어, 가림 강도와 occluder 클래스가 섞입니다.
- **victim의 label은 amodal(원본) bbox를 유지합니다.** 합성 프레임은 detector만 학습하므로
  가려진 부분까지 포함한 원래 상자를 맞히도록 두는 편이 일관됩니다.

---

## 1. 결과

### 1-1. Detector / Tracker(detector equipped) 성능 향상

- split: train `0000 0001 0002 0003 0004 0005 0006 0007 0010 0014 0015 0016`,
  eval `0008 0009 0011 0012 0013 0017 0018 0019 0020`
- baseline = 원본 KITTI train 3,644장, treatment = baseline + pasted 2,194장 = 5,838장
- 두 arm 모두 COCO-pretrained YOLOX-X를 60 epoch fine-tuning, **seed 0**, EMA checkpoint ([ByteTrack](https://github.com/FoundationVision/ByteTrack) fine tuning 관례)
- eval 4,364장 동일, 지표는 tracking과 동일한 **car/person 2-class** 기준

**Detector (COCO bbox AP, IoU 0.50:0.95, %)**

| | baseline | treatment | 차이 (pp) |
|---|---:|---:|---:|
| AP | 46.71 | 48.11 | **+1.40** |
| Occlusion AP | 30.28 | 32.34 | **+2.07** |
| Non-occlusion AP | 56.14 | 56.65 | **+0.51** |

**Tracker (BoxMOT ByteTrack, TrackEval, %)**

| | baseline | treatment | 차이 (pp) |
|---|---:|---:|---:|
| HOTA | 56.19 | 57.36 | **+1.17** |
| DetA | 51.63 | 54.14 | **+2.51** |
| AssA | 61.83 | 61.73 | **-0.11** |
| MOTA | 60.16 | 63.47 | **+3.31** |
| IDF1 | 72.59 | 73.74 | **+1.15** |

이득은 detection 쪽(DetA +2.51)에서 나오고 association(AssA)은 거의 변하지 않습니다.
합성 프레임은 detector만 학습시키므로 예상되는 방향입니다.
차이는 반올림 전 값으로 계산했습니다.

지표 정의:

- **Occlusion AP** — KITTI GT 정보 중 `occluded`가 1(partly) 또는 2(largely)인 GT만 남겨 같은 COCO
  evaluator를 다시 돌린 값. **Non-occlusion AP**는 `occluded == 0`만 남긴 값.
  `occluded == 3`(unknown)은 전체 AP에는 포함하되 두 subset에서는 제외합니다.

### 1-2. Tracklet 품질: SAM3 crop 정책

붙일 객체를 SAM3로 분할할 때 **얼마나 넓은 맥락을 보여줄지**와 **최종 mask를 어떤 상자로
자를지**를 분리해 비교했습니다. 품질은 segmentation GT가 있는 **KITTI MOTS**로 측정합니다.

- split은 위와 다르게 MOTS 원 논문을 따릅니다: train
  `0000 0001 0003 0004 0005 0009 0011 0012 0015 0017 0019 0020`
- 각 행은 그 설정에서 뽑힌 200개 tracklet에 **동일한 confidence >= 0.60 필터**를 적용한 뒤,
  MOTS GT와 매칭된 프레임만 픽셀 단위로 채점한 macro 평균입니다.
- Tracklet 생성 단계에서 MOTS GT는 선택·필터링에 전혀 쓰이지 않습니다(평가 전용).

| crop 정책 | 채택 tracklet | 채택 프레임 | MOTS 매칭 | Precision (%) | Recall (%) |
|---|---:|---:|---:|---:|---:|
| 0% padding | 163 | 8,490 | 5,476 | 97.27 | 85.13 |
| 10% padding | 169 | 8,941 | 6,021 | 97.12 | 88.41 |
| 20% padding | 162 | 8,486 | 5,784 | 97.38 | 88.74 |
| **20% padding → detector bbox 재-crop** | **162** | **8,486** | **5,784** | **97.54** | 86.66 |

채택 정책은 마지막 행입니다. padding으로 맥락을 주면 SAM3가 객체를 더 온전히 찾아내지만
(recall/IoU 상승), 최종 크기는 detector의 bbox를 믿는 편이 **mask precision**이 가장 높습니다.
즉 **분할은 context-aware하게, 크기는 detector에 더 신뢰를 주는 방향** 으로 채택하였습니다. 
이 정책으로 만든 pool이 §1-1의 학습에 그대로 쓰였습니다.

---

## 2. 환경 구성

버전 제약이 서로 호환되지 않아 환경을 셋으로 분리합니다.

| 환경 | 역할 | 정의 |
|---|---|---|
| `kds-occlusion` | 합성, YOLOX 학습, tracking/detection 평가 | [`environment.yml`](environment.yml), [`requirements.txt`](requirements.txt) |
| `kds-codetr` | legacy MMCV/MMDetection 기반 Co-DETR 추론 | [`environment.codetr.yml`](environment.codetr.yml), [가이드](docs/CODETR_WORKFLOW.md) |
| `kds-sam3` | 공식 SAM3 추론 | [`environment.sam3.yml`](environment.sam3.yml), [가이드](docs/SAM3_WORKFLOW.md) |

`.condarc`의 `envs_dirs`/`pkgs_dirs`는 절대경로이므로 새 머신에서는 먼저 저장소 경로에 맞게
수정합니다. CUDA 버전은 서버 driver에 맞추십시오.

```bash
REPO=/path/to/Occlusion_renderer
BYTETRACK=/path/to/ByteTrack       # https://github.com/ifzhang/ByteTrack
BOXMOT=/path/to/boxmot             # https://github.com/mikel-brostrom/boxmot

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

이후에는 어느 디렉터리에서든 다음으로 기본 환경을 활성화합니다.

```bash
source /path/to/Occlusion_renderer/scripts/activate_kds.sh
```

Co-DETR(Python 3.8 / PyTorch 1.11 / MMCV 1.5.0)와 SAM3(Python 3.12 / PyTorch cu128)는
위 표의 각 가이드를 따라 별도로 설치합니다. SAM3·Co-DETR가 필요한 명령은 `kds-occlusion`
에서 실행해도 해당 환경의 Python으로 자동 재실행됩니다.

### COCO-pretrained YOLOX-X

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
(Megvii-BaseDetection, Apache-2.0)가 공개한 COCO-pretrained `yolox_x.pth`에서 학습 시작, fine-tuning의
초기값으로만 사용합니다. 파일을 `$KDS_BYTETRACK_REPO/pretrained/`에 두는 것은
[ByteTrack](https://github.com/FoundationVision/ByteTrack)(Yifu Zhang, MIT)의 관례를
따른 것입니다.

보고된 모든 run이 출발한 파일은 YOLOX
[0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) asset
(md5 `c58e4a3a710e3a464bf472aaf6f4891a`)이고, 아래 명령이 그 파일을 받아 md5까지 검증합니다.

```bash
kds-prepare-pretrained          # $KDS_BYTETRACK_REPO/pretrained/yolox_x.pth
```

---

## 3. 데이터와 경로

데이터셋 세 가지가 필요합니다. 기본값은 모두 저장소 안 `datasets/` 아래이고, 다른 곳에
두었다면 §3-3의 환경변수로 덮어쓰면 됩니다.

| 데이터셋 | 쓰이는 곳 | 내려받을 용량 |
|---|---|---:|
| KITTI Tracking (left color images + training labels) | §1-1 학습·평가 전체, occluder 추출 | 약 15 GB |
| MOT17 | occluder pool 보강 (train의 FRCNN view만) | 약 5.5 GB |
| KITTI MOTS instances | §1-2 tracklet 품질 채점 전용 | 약 22 MB |

§1-1만 재현한다면 KITTI Tracking과 MOT17 둘이면 되고, MOTS는 §1-2에만 필요합니다.

### 3-1. 내려받기

아래 명령은 모두 저장소 루트에서 실행한다고 가정합니다 (`REPO=/path/to/Occlusion_renderer`).

**KITTI Tracking** — [KITTI 데이터셋 홈](https://www.cvlibs.net/datasets/kitti/)의
[tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) 페이지에서
*"Download left color images of tracking data set"* 와 *"Download training labels of tracking
data set"* 두 개만 받으면 됩니다. 브라우저 대신 직접 받으려면:

```bash
mkdir -p "$REPO/datasets/KITTI" && cd "$REPO/datasets/KITTI"
curl -LO https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
curl -LO https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
unzip -q data_tracking_image_2.zip    # -> training/image_02, testing/image_02
unzip -q data_tracking_label_2.zip    # -> training/label_02
```

라벨이 공개된 `training/`의 21개 시퀀스(`0000`~`0020`)만 씁니다. 이를 train/eval로 나눈 것이
§1-1과 §1-2의 split이며, `testing/`은 전혀 쓰지 않으므로 지워도 됩니다.

**MOT17** — [MOTChallenge](https://motchallenge.net/)의 MOT17 배포본입니다.

```bash
cd "$REPO/datasets"
curl -LO https://motchallenge.net/data/MOT17.zip
unzip -q MOT17.zip                    # -> MOT17/train, MOT17/test
```

압축을 푼 결과가 `datasets/MOT17/train/MOT17-02-FRCNN/...` 형태여야 합니다. `train/`이 바로
나왔다면 `datasets/MOT17/` 아래로 옮기십시오. 이 파이프라인은 MOT17의 GT를 읽지 않고
`train/MOT17-*-FRCNN/img1`의 프레임만 30 fps → 10 fps로 재샘플해 occluder 후보로 씁니다.
프레임 간격과 확장자를 각 시퀀스의 `seqinfo.ini`에서 읽으므로 폴더 구조를 그대로 두십시오.

**KITTI MOTS** (§1-2 전용) — KITTI 이미지 위에 segmentation 주석을 얹은 별도 배포본으로,
RWTH Aachen [MOTS 페이지](https://www.vision.rwth-aachen.de/page/mots)에서 받습니다.

```bash
mkdir -p "$REPO/datasets/MOTS_KITTI" && cd "$REPO/datasets/MOTS_KITTI"
curl -LO https://www.vision.rwth-aachen.de/media/resource_files/instances.zip
unzip -q instances.zip                # -> instances/0000/000000.png ...
```

이미지는 위에서 받은 KITTI Tracking 것을 그대로 쓰므로 주석(`instances/`)만 있으면 됩니다.
MOTS 주석은 CC BY-NC-SA 3.0이며 Voigtlaender et al., *MOTS* (CVPR 2019)와 Geiger et al.,
*KITTI* (CVPR 2012) 인용을 요구합니다. KITTI 본체 역시 비상업적 사용 조건이므로 각 배포처의
약관을 확인하십시오.

### 3-2. 최종 배치

```text
datasets/
  KITTI/training/image_02/{0000..0020}/*.png     # KITTI Tracking 이미지
  KITTI/training/label_02/{0000..0020}.txt       # KITTI Tracking 라벨
  MOT17/train/MOT17-XX-FRCNN/{img1/, seqinfo.ini}
  MOTS_KITTI/instances/{0000..0020}/*.png        # §1-2 전용
third_party/Co-DETR                              # Co-DETR 소스
third_party/sam3                                 # SAM3 소스
weights/codetr/pytorch_model.pth                 # Co-DETR ViT-L
weights/sam3/sam3.pt                             # SAM3
```

준비가 끝나면 해석된 경로를 먼저 점검합니다. 빠진 파일이 있으면 그 자리에서 알려 줍니다.

```bash
kds-preflight --config configs/default.yaml
kds-preflight --config configs/sam3_context_ablation.yaml   # MOTS 주석까지 확인
```

### 3-3. 다른 위치에 둔 경우

[`configs/base.yaml`](configs/base.yaml)의 `paths.*`가 모든 경로의 기준이고, 아래 환경변수로
파일을 옮기지 않고 덮어쓸 수 있습니다.

| 환경변수 | 용도 | 기본값 |
|---|---|---|
| `KDS_KITTI` | KITTI Tracking 루트 | `datasets/KITTI` |
| `KDS_MOT17` | MOT17 루트 | `datasets/MOT17` |
| `KDS_BYTETRACK_REPO` | ByteTrack/YOLOX checkout | `~/ByteTrack` |
| `KDS_BOXMOT_REPO` | BoxMOT checkout | `~/boxmot` |
| `KDS_TRACKEVAL_REPO` | TrackEval checkout | `~/TrackEval` |
| `KDS_YOLOX_X_CHECKPOINT` | COCO-pretrained YOLOX-X | `$KDS_BYTETRACK_REPO/pretrained/yolox_x.pth` |

```bash
export KDS_KITTI=/mnt/datasets/KITTI
export KDS_MOT17=/mnt/datasets/MOT17
```

KITTI MOTS 주석 경로만 환경변수가 아니라
[`configs/sam3_context_ablation.yaml`](configs/sam3_context_ablation.yaml)의
`sam3_context_ablation.mots_root`에서 지정합니다.

### 설정 파일 체인

각 config는 `base:` 로 상위 설정을 상속하며, 하위 파일은 바뀌는 부분만 적습니다.

```text
base.yaml                 경로 · 클래스 · GPU 예산
 └ phase1_kitti.yaml      split A, 합성/학습/추적 파라미터
    └ phase1_detector.yaml        Co-DETR + SAM3 tracklet pool (필터 없음)
       └ phase2_detector_conf.yaml   confidence >= 0.60 필터
          └ default.yaml               ← §1-1 재현의 기본 설정
             ├ phase4_detection_eval.yaml   seed 0 detector 평가
             └ official_split.yaml          MOTS 논문 split (§1-2 입력 전용)
                └ sam3_context_ablation{,_00pct,_10pct}.yaml
```

---

## 4. 재현

아래 예시의 GPU 번호는 실제 유휴 GPU로 바꾸십시오.

### 4-1. Detector / Tracker 결과 (§1-1)

```bash
# (1) split 고정. crop 허용 sequence가 eval과 겹치면 즉시 실패합니다.
kds-phase1-split --config configs/default.yaml

# (2) Co-DETR 추론 -> 환경 중립 JSON (KITTI는 train split만, MOT17은 FRCNN만)
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/default.yaml --dataset kitti
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/default.yaml --dataset mot17

# (3) ByteTrack 후보만 먼저 점검 (SAM3 미로드)
kds-build-detector-tracklets --config configs/default.yaml --inventory-only

# (4) detector bbox -> SAM3 RGBA -> confidence 필터 자동 실행.
#     shard는 seed로 정해진 원래 후보 순서로 병합되므로 GPU 스케줄링이 선택을 바꾸지 않습니다.
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-build-detector-tracklets --config configs/default.yaml --workers 4

# (5) 사건 단위 합성 -> A/B 학습셋
kds-synth-events --config configs/default.yaml
kds-build-ab --config configs/default.yaml --paste-mode append

# (6) 학습 (global batch size 4, 60 epoch, 800x1440)
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-train-yolox --config configs/default.yaml \
  --condition baseline --aug full --seed 0
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-train-yolox --config configs/default.yaml \
  --condition treatment --paste-mode append --tag conf60 --aug full --seed 0

# (7) tracking 평가 -> TrackEval, 그리고 2-class 표 생성
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-eval-boxmot --config configs/default.yaml \
  --condition baseline --epoch ep60 --seed 0 --workers 4
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-eval-boxmot --config configs/default.yaml \
  --condition treatment --paste-mode append --tag conf60 --epoch ep60 --seed 0 --workers 4
kds-aggregate --config configs/default.yaml --classes car person \
  --runs phase1_baseline_full_seed0_ep60 phase1_treatment_append_conf60_full_seed0_ep60 \
  --out artifacts/phase2/tracking_conf60/report_seed0_2class.md

# (8) detection 평가 -> 4-class로 추론 후 car/person으로 재집계
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-eval-detection-metrics \
  --config configs/phase4_detection_eval.yaml --workers 4
kds-reaggregate-detection-classes --classes car person \
  --summary artifacts/phase4/detection_metrics_random_mid/summary.json \
  --output-dir artifacts/phase4/detection_metrics_random_mid_2class
```

§1-1의 두 표는 각각 다음 파일에 그대로 남습니다.

- tracking: `artifacts/phase2/tracking_conf60/report_seed0_2class.md`
- detection: `artifacts/phase4/detection_metrics_random_mid_2class/report.md`

학습(6)을 건너뛰고 기존 checkpoint만 다시 채점할 수도 있습니다. (7)은 config의
`train.output_dir` 아래에서 `phase1_{condition}[_{paste_mode}][_{tag}]_{aug}_seed{seed}`
디렉터리를 찾고, (8)은 [`configs/phase4_detection_eval.yaml`](configs/phase4_detection_eval.yaml)의
`runs.*.checkpoint`를 그대로 읽습니다. 기존 checkpoint가 다른 곳에 있다면 이 두 경로만
맞춰 주면 됩니다.

`--seed`는 학습 seed입니다. 보고 값은 처음 실행한 seed 0 하나이며, 다른 seed로 돌리면
수치는 달라집니다.

### 4-2. Tracklet 품질 결과 (§1-2)

MOTS 논문 split의 Co-DETR detection과 ByteTrack 후보가 먼저 필요합니다.

```bash
# (1) split + Co-DETR + 후보 + 채택 정책(20% -> 재-crop -> conf 0.60) pool
kds-phase1-split --config configs/official_split.yaml
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/official_split.yaml --dataset kitti
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/official_split.yaml --dataset mot17
kds-build-detector-tracklets --config configs/official_split.yaml --inventory-only
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-build-detector-tracklets \
  --config configs/official_split.yaml --workers 4

# (2) padding 0/10/20%로 SAM3를 다시 돌리고 MOTS로 채점
bash scripts/run_sam3_crop_policy_ablation.sh
```

(2)는 padding당 SAM3를 한 번만 돌려 두 crop 정책을 동시에 만듭니다
(`variant_b_context_preserved` = padding 전체를 덮는 mask, `variant_a_bbox_clipped` =
같은 mask를 detector bbox로 재-crop). 결과 표는
`artifacts/official_split/sam3_padding_selected_200_conf60_mots/report.md`에 기록되며,
20%+재-crop 행이 (1)에서 만든 운영 pool과 **채택 프레임까지 동일한지** 함께 검증합니다.

---

## 5. 저장소 구조

| 경로 | 역할 |
|---|---|
| [`data/detection_sources.py`](data/detection_sources.py) | GT 없이 KITTI train/MOT17 프레임 열거 |
| [`mining/codetr.py`](mining/codetr.py) | Co-DETR 추론 → 환경 중립 JSON |
| [`mining/detected_tracklets.py`](mining/detected_tracklets.py) | ByteTrack association (ReID 없음) |
| [`pool/crops.py`](pool/crops.py) | context crop과 SAM3 인스턴스 매칭 |
| [`pool/build_detector_tracklet_pool.py`](pool/build_detector_tracklet_pool.py) | detector bbox → SAM3 RGBA pool |
| [`pool/filter_detector_tracklet_pool.py`](pool/filter_detector_tracklet_pool.py) | confidence 0.60 최장 연속 구간 필터 |
| [`synth/scheduler.py`](synth/scheduler.py) | victim 선정과 사건 스케줄링 |
| [`synth/geometry.py`](synth/geometry.py), [`synth/placement.py`](synth/placement.py) | mask 적분영상 기반 `rho` 탐색 |
| [`synth/paste_jitter.py`](synth/paste_jitter.py) | 프레임별 appearance jitter |
| [`synth/event_pipeline.py`](synth/event_pipeline.py) | 합성 프레임·label·event·QC 생성 |
| [`data/build_phase1.py`](data/build_phase1.py) | baseline/treatment/eval COCO JSON |
| [`train/run.py`](train/run.py), [`train/yolox_x_kitti.py`](train/yolox_x_kitti.py) | YOLOX-X 학습 |
| [`eval/run_boxmot.py`](eval/run_boxmot.py), [`eval/aggregate.py`](eval/aggregate.py) | tracking 평가와 클래스 집계 |
| [`eval/run_detection_metrics.py`](eval/run_detection_metrics.py), [`eval/reaggregate_detection_classes.py`](eval/reaggregate_detection_classes.py) | occlusion별 detection metric |
| [`eval/compare_sam3_crop_policies.py`](eval/compare_sam3_crop_policies.py), [`eval/evaluate_selected_tracklet_mots.py`](eval/evaluate_selected_tracklet_mots.py) | crop 정책 비교와 MOTS 채점 |

테스트는 표준 라이브러리만 씁니다.

```bash
source scripts/activate_kds.sh
python -m unittest discover -s tests -t tests -p "test_*.py"
```

---

## 6. 외부 자산과 라이선스

이 저장소의 코드 외에 아래 데이터·모델·라이브러리를 그대로 가져다 씁니다. 어느 것도 여기서
재배포하지 않고 §2·§3의 안내대로 각 배포처에서 직접 받도록 합니다.

| 자산 | 이 파이프라인에서의 역할 | 라이선스 |
|---|---|---|
| [KITTI Tracking](https://www.cvlibs.net/datasets/kitti/) 이미지·라벨 | 학습·평가 데이터, occluder 추출 원본 | CC BY-NC-SA 3.0 · Geiger et al., CVPR 2012 |
| [KITTI MOTS](https://www.vision.rwth-aachen.de/page/mots) `instances` | §1-2 mask 품질 채점 | CC BY-NC-SA 3.0 · Voigtlaender et al., CVPR 2019 |
| [MOT17](https://motchallenge.net/) | occluder pool 보강 (FRCNN view) | MOTChallenge 배포 조건 · Milan et al., 2016 |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) `yolox_x.pth` | fine-tuning 초기 가중치 | Apache-2.0 · Megvii |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) · [TrackEval](https://github.com/JonathonLuiten/TrackEval) · [Co-DETR](https://github.com/Sense-X/Co-DETR) | YOLOX 학습 하네스 · HOTA/CLEAR 평가 · detector | MIT |
| [BoxMOT](https://github.com/mikel-brostrom/boxmot) | ByteTrack association 구현 (pool 생성·tracking 평가) | **AGPL-3.0** |
| [SAM3](https://github.com/facebookresearch/sam3) | occluder mask 분할 | Meta SAM License |
