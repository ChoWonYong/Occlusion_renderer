# Occlusion Renderer for autonomous driving
SAM3로 분할한 동일 identity의 연속 RGBA tracklet(MOT17 human · KITTI car/human)을 KITTI
Tracking 학습 시퀀스에 시간적으로 끊김 없이 copy-paste하여 **연속적인 occlusion**을
만들고, 자동 생성한 확장 GT로 COCO-pretrained YOLOX-X detector를 fine-tuning한 뒤 동일
ByteTrack(from BoxMOT)으로 원본 KITTI eval 시퀀스에서 tracking 강건성을 검증합니다.

---

## Results (KITTI half train/half valid based on MOT protocol)

`--aug full`, ep60 EMA, seed 0/1/2 평균 ± 표준편차(sample std, ddof=1).

| Detector | 학습셋 | iters/epoch | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (원본 KITTI train, n=3) | 3,644 | 911 | 55.85 ± 0.12 | 51.32 ± 0.45 | 61.48 ± 0.72 | 59.18 ± 0.65 | 71.46 ± 0.18 |
| Ours — `replace`, paste jitter off (n=3) | 3,644 | 911 | 56.30 ± 0.43 | 52.15 ± 0.39 | 61.60 ± 0.53 | 60.15 ± 0.73 | 71.90 ± 0.56 |
| Ours — `replace` + jitter `low` (n=3) | 3,644 | 911 | 56.84 ± 0.33 | 52.79 ± 0.09 | 62.07 ± 0.76 | 60.85 ± 0.12 | 72.52 ± 0.51 |
| **Ours — `replace` + jitter `mid` (기본값, n=3)** | 3,644 | 911 | **57.19 ± 0.20** | **53.10 ± 0.01** | **62.34 ± 0.43** | **61.28 ± 0.10** | **73.05 ± 0.32** |
| Ours — `replace` + jitter `high` (n=3) | 3,644 | 911 | 56.42 ± 0.02 | 52.35 ± 0.44 | 61.59 ± 0.52 | 60.60 ± 0.66 | 72.26 ± 0.25 |
| Ours — `append` (n=1) | 5,744 | 1,436 | 55.51 | 51.46 | 60.59 | 59.04 | 70.93 |



---

## 1. Detail

- 개별 crop을 매 프레임 교체하지 않고 **동일 occluder identity의 tracklet**을 연속 paste.
- tracklet 길이는 **target 10FPS 기준 30~100프레임 가변**(min 3초). MOT17은 source FPS(14/25/30)를
  10FPS로 resampling. visibility>=0.8은 샘플 프레임에서 검사(인접 프레임 대체 허용).
- SAM3 실패 시 tracklet 전체 폐기가 아니라 **가장 긴 통과 sub-run(>=30프레임)** 을 보존.


- occluder 클래스 = **car/person 2-class (실측 65/35 재정규화)**. truck·bicycle은 KITTI train에
  30프레임 clean 연속 run이 극히 적어(truck 0개, bicycle 1개) occluder pool에서 제외합니다.
  단 detector head는 4-class(car/truck/person/bicycle)를 유지합니다(eval GT에 존재).
- **rho = occluder 실제 마스크가 victim bbox를 덮은 비율**, crop별 integral image로 계산합니다.
  victim의 amodal 마스크가 채워진 사각형이라 사각형 질의로 값이 나옵니다. 
- **크기는 난이도 손잡이가 아닙니다.** occluder 높이 = `victim높이 × factor[occluder]/factor[victim]`,
  이때 factor는 클래스의 실제 크기 분포(`class_height_range`, car 1.5 m = 1.0)에서 매 사건
  샘플링합니다.
- mild/moderate/heavy 밴드는 **목표가 아니라 gate**입니다. 물리 파라미터를 뽑고, 결과가
  완결된 사건(0→peak→0, rho>=0.1이 8–20프레임, 단일 peak, peak>=0.20)이면 채택하고 밴드는
  사후에 라벨로 붙입니다. 클래스쌍별 난이도 비대칭(보행자는 차를 heavily 가릴 수 없음)은
  실제 물리이므로 교정하지 않습니다.
- peak rho 상한은 **0.90**입니다. 이 합성 데이터는 **detector만** 학습시키고 ByteTrack은
  학습되지 않으므로, 보이는 픽셀이 0인 프레임은 tracking 난이도가 아니라 `amodal_original`
  하에서 "증거 0으로 전체 박스 예측"이라는 학습 불가능한 타깃입니다. 
- occluder identity는 **전역 최대 2회 + 같은 배경 시퀀스 내 재사용 금지**입니다.
- large-first paste, hard paste(`blend_method: none`).
- victim detector bbox 정책 = **amodal_original**(원본 bbox 유지, visible_bbox는 별도 저장).
- **붙이는 crop에는 프레임마다 다른 jitter가 걸립니다**(`paste_jitter`, 기본 `mid`). color ·
  brightness · contrast · sharpness · horizontal flip · scale · rotation을 프레임 단위로 독립
  추첨합니다. 합성 프레임은 **detector만** 학습시키므로(ByteTrack은 이 데이터로 학습되지 않음)
  프레임 간 외형이 튀어도 학습이 쓰는 신호는 손상되지 않고, 대신 pool의 62개 identity를
  픽셀 단위로 외우는 것을 막습니다. 측정값은 위 Results 참고(HOTA +0.89 vs jitter off).

---

## 2. Environments

두 환경을 분리하고, 둘 다 레포 로컬 `.conda-envs/`에 만듭니다(경로는 `.condarc`의 `envs_dirs`).
- `kds-occlusion` — ByteTrack/YOLOX, 데이터 변환, 합성, 라벨링, 평가
- `kds-sam3` — 공식 SAM3 추론 (Python 3.12, NumPy 1.26 계열)

### 2.1 활성화 (작업할 때마다)

```bash
source /path/to/Occlusion_renderer/scripts/activate_kds.sh   # 작업 디렉토리 무관
```

`conda activate kds-occlusion`을 그냥 실행하면 `Could not find conda environment`로 실패합니다.
환경 경로가 레포 `.condarc`에만 선언돼 있는데 conda가 이 파일을 자동으로 읽지 않기 때문입니다.
위 스크립트가 conda 셸 함수 로딩과 `CONDARC` export를 대신 처리합니다.

SAM3 환경은 전용 스크립트가 없으므로 prefix 경로로 활성화합니다.

```bash
conda activate /path/to/Occlusion_renderer/.conda-envs/kds-sam3
```

### 2.2 최초 설치 (환경당 1회)

`.condarc`의 `envs_dirs`·`pkgs_dirs`는 절대경로이므로, 새 머신에서는 먼저 이 레포 경로에 맞게 고칩니다.

```bash
REPO=/path/to/Occlusion_renderer
BYTETRACK=/path/to/ByteTrack      # configs/base.yaml의 paths.bytetrack_repo
BOXMOT=/path/to/boxmot            # configs/base.yaml의 paths.boxmot_repo

cd "$REPO"
export CONDARC="$REPO/.condarc"
conda env create -f environment.yml           # kds-occlusion
source scripts/activate_kds.sh

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118      # 서버 드라이버에 맞게 조정
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install -v -e "$BYTETRACK" --no-build-isolation
python -m pip install -e "$BOXMOT"
python -m pip install -e . --no-build-isolation --no-deps
```

설치가 끝난 환경에 이 블록을 다시 돌리지 마세요. `conda env create`는 prefix가 이미 있어 실패하고,
pip 줄들은 ByteTrack C 확장을 다시 빌드합니다. 활성화만 필요하면 2.1을 쓰면 됩니다.

SAM3 환경(`environment.sam3.yml`) 설치와 smoke 절차는 [docs/SAM3_WORKFLOW.md](docs/SAM3_WORKFLOW.md)에 기록되어 있습니다.

---

## 3. Dataset · dependency · checkpoint download

경로는 `configs/base.yaml`의 `paths.*`에서 지정합니다. 레포 안에 들어가는 것(KITTI, SAM3 체크포인트 등)은
이 파일 기준 상대경로로 이미 잡혀 있고, 레포 밖에 두는 외부 체크아웃·데이터셋은 `${VAR:-fallback}` 형태라
파일을 고치는 대신 환경변수로 덮어쓸 수 있습니다.

| 환경변수 | `paths` 키 | 미설정 시 기본값 |
| --- | --- | --- |
| `KDS_MOT17` | `mot17` | `~/tmp_SwapPatch/data/MOT17` |
| `KDS_BYTETRACK_REPO` | `bytetrack_repo` | `~/ByteTrack` |
| `KDS_BOXMOT_REPO` | `boxmot_repo` | `~/boxmot` |
| `KDS_TRACKEVAL_REPO` | `trackeval_repo` | `~/BankTweak/TrackEval` |
| `KDS_YOLOX_X_CHECKPOINT` | `coco_pretrained_yolox_x` | `$KDS_BYTETRACK_REPO/pretrained/yolox_x.pth` |

`KDS_BYTETRACK_REPO`만 바꾸면 YOLOX-X 체크포인트 경로도 같이 따라갑니다. 체크포인트를 다른 곳에 두었을 때만
`KDS_YOLOX_X_CHECKPOINT`를 별도로 지정하면 됩니다.

```bash
export KDS_MOT17=/data/MOT17          # 예: 공용 스토리지에 있을 때
python scripts/preflight.py --config configs/phase1_kitti.yaml   # 해석된 경로 전부 점검
```

### 3.1 KITTI Tracking (배경/GT/eval)
[KITTI Tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)에서 Download left color images of tracking data set, Download training labels of tracking data set을 받아 아래 구조로 풉니다.

```bash
unzip -n datasets/KITTI/data_tracking_image_2.zip -d datasets/KITTI
unzip -n datasets/KITTI/data_tracking_label_2.zip -d datasets/KITTI
# datasets/KITTI/training/{image_02/0000..0020, label_02/0000.txt..0020.txt}
```

### 3.2 MOT17 (human tracklet source)
[MOTChallenge MOT17](https://www.codabench.org/competitions/10049/#/pages-tab) train을 받아 풀고
`KDS_MOT17`을 그 경로로 지정합니다(`train/MOT17-XX-FRCNN/{img1,gt,seqinfo.ini}`).
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
git clone https://github.com/ifzhang/ByteTrack   # KDS_BYTETRACK_REPO
# COCO-pretrained YOLOX-X 가중치(yolox_x.pth)는 YOLOX 공식 model zoo
# (Megvii-BaseDetection/YOLOX) releases에서 받아 ByteTrack/pretrained/yolox_x.pth에 둡니다.
# (KDS_YOLOX_X_CHECKPOINT)

# BoxMOT (ByteTrack 추적기 구현)
git clone https://github.com/mikel-brostrom/boxmot  # KDS_BOXMOT_REPO

# TrackEval (HOTA/CLEAR/Identity)
git clone https://github.com/JonathonLuiten/TrackEval  # KDS_TRACKEVAL_REPO
```

기본값(홈 디렉토리 바로 아래)과 다른 곳에 clone했다면 3절 앞머리의 환경변수로 지정하고,
`paths.{sam3_checkpoint, kitti_tracking}`처럼 레포 안에 두는 항목만 `configs/base.yaml`에서 직접 맞춥니다.

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

paste jitter 강도별 arm은 파생 config로 돌립니다. 각각 자기 출력 디렉토리를 쓰므로 이미 학습한
arm의 입력을 덮지 않습니다. 나머지 설정은 전부 `phase1_kitti.yaml`에서 상속합니다(`base:` 키).

```bash
kds-synth-events --config configs/phase1_kitti_nojitter.yaml       # jitter off (대조군)
kds-synth-events --config configs/phase1_kitti_pastejit_low.yaml   # 약함
kds-synth-events --config configs/phase1_kitti_pastejit_high.yaml  # 강함
```

> `phase1_kitti.yaml`은 기본이 `mid`이므로 위 명령을 그 config로 돌리면
> `artifacts/phase1/tracklet_synthetic/`이 jitter 적용본으로 덮입니다. jitter off 데이터가
> 필요하면 `phase1_kitti_nojitter.yaml`로 재생성하세요(시드 고정이라 동일 결과).

### 4.4 A/B 데이터셋
```bash
kds-build-ab --config configs/phase1_kitti.yaml                        # replace (default)
kds-build-ab --config configs/phase1_kitti.yaml --paste-mode append    # 비교용
```
- **Baseline** = 원본 KITTI train split 그대로. copy-paste가 방법론 자체이므로 두 arm은
  그것만 다르고, mosaic/mixup/flip/HSV 등 온라인 증강은 학습기가 양쪽에 동일하게 적용합니다.
  (이전의 equal-budget flip/jitter 패딩은 제거 — 이미지 수를 맞추려던 것이지만 baseline을
  "증강 없음"이 아니라 "다른 증강"으로 만들었습니다.)
- **Treatment `replace`** (기본값) = paste 프레임이 원본 프레임을 **교체**. 두 arm의 이미지
  수와 iteration 예산이 같아지고 픽셀 내용만 달라집니다. 교체 단위는 시간적으로 겹치는
  tracklet들의 **연결요소**입니다 — 렌더된 프레임은 그 시점의 모든 occluder를 함께
  굽기 때문에 tracklet 하나만 끄려면 재렌더가 필요하고, 프레임 단위로 동전을 던지면
  이 방법이 만들려는 시간적 연속성이 깨집니다.
- **Treatment `append`** = 원본 train + paste 프레임. 이미지와 gradient step이 1.6배로
  늘지만 같은 배경이 clean/pasted 두 벌로 들어갑니다. 측정 결과 baseline보다 낮아
  (HOTA 55.12 vs 55.39 ± 0.10) 기본값에서 제외했고, 비교 조건으로만 남겨둡니다.
- 산출 JSON: `artifacts/phase1/yolox_data/annotations/{baseline_train,treatment_replace_train,treatment_append_train,eval,kitti_train}.json`

### 4.5 detector fine-tuning (YOLOX-X, COCO에서)
```bash
# condition ∈ {kitti(원본만), baseline(원본만), treatment(+paste)}
# aug ∈ {none(letterbox+normalize만), nofliphue(brightness/contrast/saturation),
#        jitter(+flip +hue), full(+mosaic/mixup/affine, 마지막 10ep off)}
# multi-scale resize는 학습기가 구동하므로 어느 레벨에서도 켜져 있습니다.
# --paste-mode 생략 시 dataset.paste_mode(=replace)를 따릅니다. 4.4 에서 빌드한 것과
# 일치해야 하며, 평가(4.6)도 같은 값으로 해석되므로 세 단계가 자동으로 맞물립니다.
CUDA_VISIBLE_DEVICES=0 kds-train-yolox --condition treatment --aug full --seed 0
CUDA_VISIBLE_DEVICES=1 kds-train-yolox --condition baseline  --aug full --seed 0
# 비교 조건: CUDA_VISIBLE_DEVICES=2 kds-train-yolox --condition treatment --paste-mode append --aug full --seed 0
# 데이터셋만 다른 arm은 --config + --tag 로 구분 (이름이 겹치면 서로를 덮어씀)
CUDA_VISIBLE_DEVICES=3 kds-train-yolox --config configs/phase1_kitti_pastejit_high.yaml \
  --condition treatment --aug full --seed 0 --tag pastejit_high
# smoke: --max-epoch 1 / 중단된 run 이어서: --resume
```
출력 `artifacts/phase1/yolox_runs/phase1_<label>[_<paste_mode>][_<tag>]_<aug>_seed<seed>/`:
`latest_ckpt.pth.tar`(=ep60, EMA), `last_mosaic_epoch_ckpt.pth.tar`(=ep59 스냅샷 — no-aug
경계에서 매 epoch 덮어써지므로 실제 내용은 마지막 epoch 진입 시점입니다).
실험 이름은 `train.run._experiment_name`과 `eval.run_boxmot._experiment_dir_name`이 동일한
규칙으로 만들므로, 학습한 run을 평가가 그대로 찾습니다.

`--resume`은 그 run의 `latest_ckpt.pth.tar`에서 이어갑니다(재시작 epoch은 체크포인트에 기록된
값). ByteTrack의 `resume_train`은 `-c`가 지정돼 있으면 그 파일을 resume 대상으로 삼기 때문에,
`--resume`을 주면 `-c`가 COCO 가중치 대신 해당 run의 체크포인트로 바뀝니다. 저장되는 `model`은
EMA 가중치이므로 재개 경계에서 raw 가중치가 EMA 값으로 한 번 덮이고 데이터 순서 RNG도 다시
흐릅니다 — 중단 없이 돈 run과 절차가 완전히 같지는 않습니다.

### 4.6 tracking evaluation (BoxMOT ByteTrack + TrackEval)
```bash
# eval 9시퀀스 원본 KITTI에서 YOLOX 추론 → ByteTrack → 클래스별 TrackEval(HOTA/CLEAR/Identity)
CUDA_VISIBLE_DEVICES=0 kds-eval-boxmot --condition treatment --aug full --epoch ep60 --seed 0
CUDA_VISIBLE_DEVICES=1 kds-eval-boxmot --condition kitti     --aug full --epoch ep60 --seed 0
# 학습에 --tag 를 줬다면 평가에도 같은 값을 줘야 체크포인트를 찾습니다
CUDA_VISIBLE_DEVICES=2 kds-eval-boxmot --config configs/phase1_kitti_pastejit_high.yaml \
  --condition treatment --aug full --epoch ep60 --seed 0 --tag pastejit_high
# 참조: COCO-pretrained
kds-eval-boxmot --model-source coco-pretrained
# 실험명 규칙 밖의 체크포인트는 --checkpoint 로 직접 지정
```
ByteTrack 파라미터는 `configs/*.yaml`의 `tracker.*`(frame_rate10 · min_conf0.10 · track_thresh0.45 ·
match_thresh0.80 · track_buffer30 · per_class · reid off). 결과는
`artifacts/phase1/tracking/<run_name>/trackers/KDS_<CLASS>-eval/<run_name>/pedestrian_{summary.txt,detailed.csv}`.

### 4.7 비교표 집계
```bash
kds-aggregate --runs \
  phase1_kitti_finetuned_full_seed{0,1,2}_ep60 \
  phase1_treatment_replace_full_seed{0,1,2}_ep60 \
  --out artifacts/phase1/results.md
```
per-class TrackEval 출력에서 **전체(combined)** 를 detection-weighted(alpha별 TP/FN/FP + TP-weighted
AssA)로 조합하고 MOTA/IDF1은 count 재계산합니다. `fine_tuned_baseline.md`(COCO
및 기존 KITTI fine-tune 기준)와 동일 산출식이라 직접 비교 가능합니다.

run 이름에서 `_seedN`을 떼어 같은 조건의 seed들을 자동으로 묶고, 개별 행 아래에
**평균 ± 표준편차**(sample std, ddof=1) 행을 덧붙입니다. n=1이면 평균만 표기합니다.

---

## 5. 주요 config 노브 (`configs/phase1_kitti.yaml`)

| 섹션 | 키 | 의미 |
|---|---|---|
| `tracklet_pool` | `target_fps, min_frames, max_frames, visibility_substitution_window` | MOT17 FPS-aware 가변길이 pool |
| `kitti_sam3_pool` | `target_fps, min_frames, max_frames, prompts` | KITTI multi-class tracklet pool |
| `tracklet_synthesis` | `class_ratio, class_height_range, max_lateral_offset_fraction, peak_rho_max, event_gate_floor, effective_event_frames, victim_events_per_100_frames, max_occluders_per_victim, victim_detector_bbox_policy` | 스케줄러/배치/rho 정책 |
| `tracklet_synthesis.paste_jitter` | `preset(off\|low\|mid\|high)` + 항목별 범위 override | 붙이는 crop의 프레임별 외형 jitter (기본 `mid`) |
| `dataset` | `synthetic_frames(paste_only\|all), paste_mode(replace\|append), paste_probability` | A/B 조립 |
| `train` | `epochs, batch_size, input_size, fp16, output_dir` | YOLOX-X 학습 |
| `tracker` / `detector` | ByteTrack · 추론 threshold | 평가 |

occlusion_level band(0.20/0.35/0.65)와 rho band(mild/moderate/heavy)는 정렬되어 있습니다.
밴드는 목표가 아니라 채택 gate이자 사후 라벨입니다.

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
synth/paste_jitter.py       # 붙이는 crop의 프레임별 외형 jitter(프리셋/패치·박스 변환)
synth/event_pipeline.py     # 통합 합성 렌더 + 확장 GT + QC
label/{compute,events}.py   # frame occlusion·bbox 정책·temporal event
data/build_phase1.py        # equal-budget baseline/treatment 조립
train/{run.py, yolox_x_kitti.py} # YOLOX-X fine-tune + aug 토글
eval/run_boxmot.py          # ByteTrack 추론 + TrackEval layout
eval/aggregate.py           # per-class → 전체 비교표
```
