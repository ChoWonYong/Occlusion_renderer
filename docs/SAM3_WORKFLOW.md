# SAM3 환경 (`kds-sam3`)

SAM3는 Python 3.12 / PyTorch(cu128)를 요구해 `kds-occlusion`(Py3.10, cu118, ByteTrack/YOLOX)과
분리합니다. SAM3는 RGBA tracklet pool 생성까지만 담당하고, 결과 PNG/JSON
(`artifacts/phase1/{mot17_sam3_tracklets,kitti_sam3_tracklets}/`)로 `kds-occlusion`과 연결됩니다.

## 1. 환경 + SAM3 소스

```bash
export CONDARC=$PWD/.condarc
conda env create --prefix ./.conda-envs/kds-sam3 -f environment.sam3.yml
./.conda-envs/kds-sam3/bin/python -m pip install torch==2.10.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git third_party/sam3
./.conda-envs/kds-sam3/bin/python -m pip install -e third_party/sam3
./.conda-envs/kds-sam3/bin/python -m pip install -e . --no-deps
```

`sam3_precision: auto`는 V100에서 FP16, Ampere+에서 BF16을 사용합니다(V100은 Flash Attention 미사용).

## 2. Gated checkpoint

<https://huggingface.co/facebook/sam3>에서 접근 승인 후:

```bash
./.conda-envs/kds-sam3/bin/hf auth login
mkdir -p weights/sam3
./.conda-envs/kds-sam3/bin/hf download facebook/sam3 sam3.pt config.json --local-dir weights/sam3
```

`configs/base.yaml`의 `paths.sam3_checkpoint`(weights/sam3/sam3.pt)와
`paths.sam3_bpe`(third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz)가 이를 가리킵니다.

## 3. Smoke (선택)

SAM3 명령은 `kds-occlusion`에서 실행해도 자동으로 `kds-sam3` Python으로 재실행됩니다.

```bash
# SAM3 미로드 후보/leakage 점검
kds-build-tracklets --config configs/phase1_kitti.yaml --inventory-only
kds-build-kitti-sam3-pool --config configs/phase1_kitti.yaml --tracklet --inventory-only

# 단일 프레임 mask 품질 확인
kds-sam3-smoke --config configs/phase1_kitti.yaml --candidate-id 1 --frame-offset 15
kds-kitti-sam3-smoke --config configs/phase1_kitti.yaml --category car
```

전체 pool 생성 및 이후 합성/학습/평가는 [README](../README.md) 4절을 따릅니다.
