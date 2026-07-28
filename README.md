# Occlusion Renderer for autonomous driving
SAM3로 분할한 동일 identity의 연속 RGBA tracklet(MOT17 human · KITTI car/human)을 KITTI
Tracking 학습 시퀀스에 시간적으로 끊김 없이 copy-paste하여 **연속적인 occlusion**을
만들고, 자동 생성한 확장 GT로 COCO-pretrained YOLOX-X detector를 fine-tuning한 뒤 동일
ByteTrack(from BoxMOT)으로 원본 KITTI eval 시퀀스에서 tracking 강건성을 검증합니다.

추후 tracker, detector, dataset, render 세부 사항 변경 등 성능 개선의 여지가 충분합니다.

---

## Results (KITTI half train/half valid based on MOT protocol)

> **진행 중 (2026-07-28).** 합성기와 baseline 정의가 모두 바뀌어 재학습 중이며, treatment는
> seed 0만 확정입니다 (seed 1·2 학습 중). baseline은 3 seed 완료.

seed 0, `--aug full`, ep60 EMA. baseline은 seed 0/1/2 평균 ± 표준편차(ddof=1).

| Detector | 학습셋 | iters/epoch | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (원본 KITTI train, n=3) | 3,644 | 911 | 55.39 ± 0.10 | 50.81 ± 0.45 | 61.05 ± 0.70 | 57.94 ± 0.66 | 70.99 ± 0.15 |
| **Ours — `replace` (n=1)** | 3,644 | 911 | **56.21** | **52.01** | **61.55** | **59.49** | **71.98** |
| Ours — `append` (n=1) | 5,744 | 1,436 | 55.12 | 51.06 | 60.19 | 58.02 | 70.56 |

**`replace`가 주 결과입니다**: baseline과 이미지 수·gradient step이 정확히 같고 픽셀 내용만
다릅니다. HOTA +0.82는 baseline seed 분산(±0.10)의 8배이며, 개선의 출처는 DetA(+1.20)와
FN(8,858 → 7,876)입니다 — 즉 가려진 객체를 더 찾아내는 쪽이고 association(AssA +0.50)이
아닙니다. 가림 증강의 기대 효과와 일치하는 방향입니다.

`append`(paste 프레임을 추가)는 1.6배의 학습 예산을 받고도 baseline보다 **낮습니다**(−0.27).
같은 배경이 원본 1장 + paste 1장으로 두 번 들어가 학습셋의 73%가 중복 배경이 되고,
AssA가 가장 크게 떨어집니다(61.05 → 60.19). 그래서 `dataset.paste_mode` 기본값은 `replace`입니다.

### 이전 결과와의 관계

구 README는 **+1.27**(54.42 → 55.69)로 기재했으나, 그 baseline은 원본이 아니라 **equal-budget
baseline**(원본 + flip/color-jitter 1,740장)이었습니다. 같은 구 treatment를 원본 fine-tune
기준(55.39)으로 보면 **+0.30**입니다. equal-budget 패딩이 baseline을 약 0.97 HOTA 깎고 있었고,
이것이 해당 패딩을 제거한 이유입니다.

| 비교 대상 | 구 treatment | 차이 |
|---|---:|---:|
| equal-budget baseline 54.42 | 55.69 | +1.27 |
| 원본 fine-tune baseline 55.39 | 55.69 | **+0.30** |

---

## 1. Detail

- 개별 crop을 매 프레임 교체하지 않고 **동일 occluder identity의 tracklet**을 연속 paste.
- tracklet 길이는 **target 10FPS 기준 30~100프레임 가변**(min 3초). MOT17은 source FPS(14/25/30)를
  10FPS로 resampling. visibility>=0.8은 샘플 프레임에서 검사(인접 프레임 대체 허용).
- SAM3 실패 시 tracklet 전체 폐기가 아니라 **가장 긴 통과 sub-run(>=30프레임)** 을 보존.
- occluder 클래스 = **car/person 2-class (실측 65/35 재정규화)**. truck·bicycle은 KITTI train에
  >=30프레임 clean 연속 run이 극히 적어(truck 0개, bicycle 1개) occluder pool에서 제외합니다.
  단 detector head는 4-class(car/truck/person/bicycle)를 유지합니다(eval GT에 존재).
- **rho = occluder 실제 마스크가 victim bbox를 덮은 비율**, crop별 integral image로 계산합니다.
  victim의 amodal 마스크가 채워진 사각형이라 사각형 질의로 정확값이 나오고, 비용은 bbox
  겹침과 같은 O(1)입니다. 이전의 bbox proxy는 occluder 클래스에 따라 rho를 0.40~0.72배로
  과대평가해(KITTI car 0.72 / MOT17 person 0.59 / KITTI person 0.40) 가림 강도가 occluder
  클래스와 교란되어 있었습니다.
- **크기는 난이도 손잡이가 아닙니다.** occluder 높이 = `victim높이 × factor[occluder]/factor[victim]`,
  이때 factor는 클래스의 실제 크기 분포(`class_height_range`, car 1.5 m = 1.0)에서 매 사건
  샘플링합니다. 목표 rho를 역산하던 이전 방식은 채택 배치의 69%를 물리적으로 불가능한
  크기로 렌더했습니다(car의 61%가 1.20 m, person의 36%가 2.21 m).
- **난이도는 lateral offset이 결정합니다.** offset은 `(w_occluder + w_victim)/2` — 두 박스가
  겹침을 멈추는 중심 간 거리 — 단위로 샘플링합니다. 좌우 이동은 occluder를 victim과 같은
  깊이에 유지하므로 크기 조절과 달리 물리적 타당성을 해치지 않고, 클래스쌍과 무관하게
  0(완전 정렬)~1(완전 분리) 전 구간에 도달합니다.
- mild/moderate/heavy 밴드는 **목표가 아니라 gate**입니다. 물리 파라미터를 뽑고, 결과가
  완결된 사건(0→peak→0, rho>=0.1이 8–20프레임, 단일 peak, peak>=0.20)이면 채택하고 밴드는
  사후에 라벨로 붙입니다. 클래스쌍별 난이도 비대칭(보행자는 차를 heavily 가릴 수 없음)은
  실제 물리이므로 교정하지 않습니다.
- peak rho 상한은 **0.90**입니다. 이 합성 데이터는 **detector만** 학습시키고 ByteTrack은
  학습되지 않으므로, 보이는 픽셀이 0인 프레임은 tracking 난이도가 아니라 `amodal_original`
  하에서 "증거 0으로 전체 박스 예측"이라는 학습 불가능한 타깃입니다. 상한 1.0으로 측정했을 때
  96개 사건 중 6개만 peak>0.90이었는데 그 6개가 완전가림 260프레임을 전부 만들었습니다
  (1.0에 닿은 사건은 그 상태로 오래 머무름). 상한을 두면 그 6개만 더 작은 offset으로
  재추첨되고, 사건은 여전히 0.85~0.90까지 도달합니다(이전 합성기의 실질 상한은 ~0.55).
- **baseline이 학습하는 라벨은 하나도 지우지 않습니다.** 가림 정도와 무관하게 모든 victim이
  amodal 박스를 가진 positive로 남습니다. 한때 `iscrowd=1` ignore 규칙을 넣었다가 제거했는데,
  ByteTrack의 `MOTDataset`은 `iscrowd != 0`과 `area == 0`을 목록에서 빼고 YOLOX에는 ignore
  처리가 없어서, 그 자리는 ignore가 아니라 **objectness 0인 명시적 negative**가 됩니다.
  `amodal_original` 정책의 목적과 정반대이고 baseline에 있는 라벨을 지웁니다(smoke 시퀀스
  기준 real annotation의 3.0%, 그중 15건은 rho<0.5로 약하게만 가려진 것). 같은 이유로
  `area`는 검출 타깃인 amodal 박스 기준으로 기록합니다(가시 면적은 `visible_area`에 별도 저장).
- 동시성은 **victim당** 제약입니다(`max_occluders_per_victim`). 이전의 전역 프레임 카운터는
  화면 반대편의 서로 다른 victim을 가리는 사건끼리 경쟁시켜 배치 성공분의 44%를 버렸고,
  화면당 평균 occluder를 0.52개로 만들었습니다.
- **장면 단위 gate(`SceneBudget`)가 예상치 못한 완전가림 세 경로를 전부 막습니다.** 사건
  gate는 `(occluder, 그 occluder의 victim)` 쌍만 보므로, 그것만으로는 occluder가 궤적을
  지나며 **무관한 객체**를 덮는 것을 못 막습니다(캡 초과 174프레임 중 118건이 이 경우였고
  전부 의도된 victim이 아니었습니다). `SceneBudget`은 배치 확정 전에 ① 이미 배치된 occluder와
  박스가 겹치지 않을 것, ② 프레임 내 **모든** 실제 annotation의 누적 커버리지가 캡 이하일 것을
  함께 검사합니다. ①이 ②를 정확하게 만듭니다 — occluder가 서로 겹치지 않으면 마스크가
  서로소이므로 한 객체 위의 합집합이 개별 기여의 **정확한 합**이 됩니다. 결과: 완전가림
  프레임 0, rho>0.90 프레임 0, 합성체끼리 가림 0. 대가는 사건 97→85, heavy 밴드 30→15.
- occluder identity는 **전역 최대 2회 + 같은 배경 시퀀스 내 재사용 금지**입니다.
- large-first paste, hard paste(`blend_method: none`).
- victim detector bbox 정책 = **amodal_original**(원본 bbox 유지, visible_bbox는 별도 저장).

### 산출 규모 (KITTI train 12 시퀀스)

| | 값 |
|---|---:|
| paste 사건 (occluder tracklet) | 85 (mild 37 / moderate 33 / heavy 15) |
| 합성 시퀀스 총 프레임 | 3,644 (원본과 1:1) |
| occluder가 그려진 프레임 | 2,100 (57.6%) |
| 가림이 발생한 victim-frame | 3,418 (real annotation의 18.6%) |
| 화면당 동시 occluder | 평균 0.87, 최대 4 |
| 사건당 노출 길이 | 최소 30 / 중앙값 33 / 최대 95 |

프레임당 occluder 수는 직접 설정하는 값이 아니라
`(victim_events_per_100_frames / 100) × E[노출 길이] × 채택률`로 나오며 시퀀스 길이와
무관합니다. 현재 `3.0/100 × 37.3 × 0.79 = 0.87`.

### 남아 있는 한계 (의도적으로 유지)

- 모든 사건의 peak이 노출 구간 **정중앙에 고정**되어 시간 구조가 대칭입니다.
- 노출 길이는 설계상 30–100 가변이지만 acceptance 기준(rho>=0.1 run이 단일·8–20프레임)이
  긴 노출을 체계적으로 탈락시켜 **실제 중앙값은 33프레임**입니다.
- victim의 amodal 마스크는 bbox proxy입니다(victim에는 SAM3를 쓰지 않음).
- `cap_events_to_distinct_victims`와 `one_event_per_victim`은 **현재 무효 설정**입니다.
  전자는 True/False가 같은 결과를 내고(`zip(chosen_victims, occluders)`이 이미 victim 수로
  자름; False는 occluder만 더 뽑아 identity 예산을 낭비), 후자는 어느 코드도 읽지 않습니다
  (victim 중복이 없는 건 `rng.sample`의 비복원 추출 때문). 사건 밀도를 올리려면 이 구조를
  먼저 고쳐야 합니다 — 지금은 사건 수가 `min(target, 적격 victim 수)`에서 멈춥니다.


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
# aug ∈ {none, jitter(flip+HSV), full(+mosaic/mixup, 마지막 10ep off)}
# --paste-mode 생략 시 dataset.paste_mode(=replace)를 따릅니다. 4.4 에서 빌드한 것과
# 일치해야 하며, 평가(4.6)도 같은 값으로 해석되므로 세 단계가 자동으로 맞물립니다.
CUDA_VISIBLE_DEVICES=0 kds-train-yolox --condition treatment --aug full --seed 0
CUDA_VISIBLE_DEVICES=1 kds-train-yolox --condition baseline  --aug full --seed 0
# 비교 조건: CUDA_VISIBLE_DEVICES=2 kds-train-yolox --condition treatment --paste-mode append --aug full --seed 0
# smoke: --max-epoch 1
```
출력 `artifacts/phase1/yolox_runs/phase1_<label>[_<paste_mode>]_<aug>_seed<seed>/`:
`latest_ckpt.pth.tar`(=ep60, EMA), `last_mosaic_epoch_ckpt.pth.tar`(=ep50 경계 스냅샷).
실험 이름은 `train.run._experiment_name`과 `eval.run_boxmot._experiment_dir_name`이 동일한
규칙으로 만들므로, 학습한 run을 평가가 그대로 찾습니다.

### 4.6 tracking evaluation (BoxMOT ByteTrack + TrackEval)
```bash
# eval 9시퀀스 원본 KITTI에서 YOLOX 추론 → ByteTrack → 클래스별 TrackEval(HOTA/CLEAR/Identity)
CUDA_VISIBLE_DEVICES=0 kds-eval-boxmot --condition treatment --aug full --epoch ep60 --seed 0
CUDA_VISIBLE_DEVICES=1 kds-eval-boxmot --condition kitti     --aug full --epoch ep60 --seed 0
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
synth/event_pipeline.py     # 통합 합성 렌더 + 확장 GT + QC
label/{compute,events}.py   # frame occlusion·bbox 정책·temporal event
data/build_phase1.py        # equal-budget baseline/treatment 조립
train/{run.py, yolox_x_kitti.py} # YOLOX-X fine-tune + aug 토글
eval/run_boxmot.py          # ByteTrack 추론 + TrackEval layout
eval/aggregate.py           # per-class → 전체 비교표
```
