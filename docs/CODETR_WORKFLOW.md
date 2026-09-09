# Co-DETR 환경 (`kds-codetr`)

공개된 `zongzhuofan/co-detr-vit-large-coco` checkpoint는 MMDetection 2.25.3 / MMCV 1.5.0을
쓰므로, ByteTrack·SAM3 환경과 분리해 `kds-codetr`에 둡니다. 두 환경 사이의 유일한
인터페이스는 버전이 찍힌 JSON 파일입니다.

## 1. 설치

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
```

공식 저장소가 자체 MMDetection fork를 포함합니다. **이 환경에 최신 MMDetection을 설치하지
마십시오.**

## 2. Checkpoint

```bash
mkdir -p weights/codetr
huggingface-cli download zongzhuofan/co-detr-vit-large-coco pytorch_model.pth \
  --local-dir weights/codetr
```

`configs/base.yaml`의 `paths.codetr_checkpoint`가 이 파일을 가리킵니다.

## 3. 동작 정책

- score 0.10 이상의 모든 박스를 남깁니다. ByteTrack이 `[min_conf=0.10, track_thresh=0.45)`
  구간의 낮은 점수 박스로 2차 association을 하기 때문입니다.
- 붙일 객체의 클래스는 **점수와 무관하게** Co-DETR의 최고-confidence label을 씁니다.
  confidence는 품질 게이트일 뿐 클래스를 바꾸지 않으며, adapter가 label을 아예 주지 않는
  예외에만 bbox aspect ratio를 fallback으로 씁니다.
- COCO의 `bus`/`truck` detection은 붙일 때 `car`로 합칩니다.
- confidence 0.60 필터는 pool 단계에서 프레임 단위로 적용되어 **첫 번째 최장 연속 구간**만
  남기고, 남은 길이가 30프레임 미만이면 tracklet 전체를 버립니다. GT와 identity purity는
  선택 입력이 아닙니다.

## 4. 실행

명령은 `kds-occlusion` 환경에서 실행해도 자동으로 `kds-codetr` Python으로 재실행됩니다.
전체 순서는 [README](../README.md)의 §4에 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/default.yaml --dataset kitti
CUDA_VISIBLE_DEVICES=0 kds-codetr-detect --config configs/default.yaml --dataset mot17
```

Co-DETR 추론은 GPU 1개를 씁니다. 최소 smoke test는 각 명령에
`--max-sequences 1 --max-frames-per-sequence 40`을 붙이면 됩니다.
