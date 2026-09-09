# SAM3 환경 (`kds-sam3`)

SAM3는 Python 3.12 / PyTorch(cu128)를 요구해 `kds-occlusion`(Py3.10, cu118,
ByteTrack/YOLOX)과 분리합니다. SAM3는 RGBA tracklet pool 생성까지만 담당하고, 결과
PNG/JSON으로 `kds-occlusion`과 연결됩니다.

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

`sam3_precision: auto`는 V100에서 FP16, Ampere 이상에서 BF16을 씁니다(V100은 Flash
Attention 미사용).

## 2. Gated checkpoint

<https://huggingface.co/facebook/sam3>에서 접근 승인 후:

```bash
./.conda-envs/kds-sam3/bin/hf auth login
mkdir -p weights/sam3
./.conda-envs/kds-sam3/bin/hf download facebook/sam3 sam3.pt config.json --local-dir weights/sam3
```

`configs/base.yaml`의 `paths.sam3_checkpoint`(`weights/sam3/sam3.pt`)와
`paths.sam3_bpe`(`third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`)가 이를
가리킵니다.

## 3. 실행

SAM3가 필요한 명령은 `kds-occlusion`에서 실행해도 자동으로 `kds-sam3` Python으로
재실행됩니다.

```bash
# SAM3를 올리지 않고 ByteTrack 후보만 점검
kds-build-detector-tracklets --config configs/default.yaml --inventory-only

# detector bbox -> 20% context crop -> SAM3 -> detector bbox 재-crop -> RGBA pool
CUDA_VISIBLE_DEVICES=0,1,2,3 kds-build-detector-tracklets --config configs/default.yaml --workers 4
```

shard는 seed로 정해진 원래 후보 순서대로 병합되므로 GPU 스케줄링이 최종 선택을 바꾸지
않습니다. 독립 SAM3 shard는 최대 4 GPU까지 쓸 수 있고, `CUDA_VISIBLE_DEVICES`의 개수와
`--workers`는 일치해야 합니다.

crop 정책(padding 0/10/20%, detector bbox 재-crop)을 KITTI MOTS로 비교하는 방법은
[README](../README.md)의 §4-2에 있습니다.
