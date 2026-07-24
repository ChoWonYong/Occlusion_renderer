from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from segment.base import Instance, SegBackend


class Sam3Backend(SegBackend):
    """Official Meta SAM3 image backend using open-vocabulary text prompts."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        bpe_path: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        precision: str = "auto",
    ) -> None:
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "Official SAM3 is required in the separate kds-sam3 environment. "
                "See docs/SAM3_WORKFLOW.md."
            ) from exc

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        tokenizer_path = Path(bpe_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {tokenizer_path}")
        selected_device = "cuda" if device == "auto" else device
        if selected_device != "cuda":
            raise RuntimeError("SAM3 extraction requires device=cuda in this project")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3 requires a visible CUDA GPU, but torch.cuda.is_available() is false in kds-sam3"
            )

        self._torch = torch
        self._device = selected_device
        capability_major, _ = torch.cuda.get_device_capability()
        if precision == "auto":
            # V100 (sm_70) has efficient FP16 Tensor Cores but no native BF16
            # Tensor Cores. Prefer BF16 only on Ampere (sm_80) or newer.
            self._autocast_dtype = torch.bfloat16 if capability_major >= 8 else torch.float16
        elif precision == "float16":
            self._autocast_dtype = torch.float16
        elif precision == "bfloat16":
            self._autocast_dtype = torch.bfloat16
        else:
            raise ValueError("SAM3 precision must be one of: auto, float16, bfloat16")
        self.precision = str(self._autocast_dtype).removeprefix("torch.")
        self._model = build_sam3_image_model(
            bpe_path=str(tokenizer_path),
            device=selected_device,
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            enable_segmentation=True,
        )
        self._processor = Sam3Processor(
            self._model,
            device=selected_device,
            confidence_threshold=float(confidence_threshold),
        )

    def _autocast(self):
        if self._device == "cuda":
            return self._torch.autocast("cuda", dtype=self._autocast_dtype)
        return nullcontext()

    def detect_and_mask(self, image: np.ndarray, prompts: list[str]) -> list[Instance]:
        from PIL import Image

        if len(prompts) != 1 or not str(prompts[0]).strip():
            raise ValueError("SAM3 image extraction requires exactly one non-empty text prompt")
        prompt = str(prompts[0]).strip()
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM3 input must be an RGB image")
        with self._torch.inference_mode(), self._autocast():
            state = self._processor.set_image(Image.fromarray(rgb))
            output = self._processor.set_text_prompt(state=state, prompt=prompt)

        masks = output["masks"].detach().cpu().numpy()
        boxes = output["boxes"].detach().float().cpu().numpy()
        scores = output["scores"].detach().float().cpu().numpy()
        instances: list[Instance] = []
        for raw_mask, raw_box, raw_score in zip(masks, boxes, scores):
            mask = np.asarray(raw_mask).squeeze()
            if mask.shape != rgb.shape[:2]:
                raise ValueError(f"SAM3 mask shape {mask.shape} does not match image {rgb.shape[:2]}")
            x1, y1, x2, y2 = (float(value) for value in np.asarray(raw_box).reshape(-1)[:4])
            instances.append(
                Instance(
                    bbox=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    mask=(mask != 0).astype(np.uint8),
                    category=prompt,
                    score=float(np.asarray(raw_score).reshape(-1)[0]),
                )
            )
        return instances
