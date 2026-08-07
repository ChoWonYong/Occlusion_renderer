"""YOLOX-X experiment consumed by ByteTrack's tools/train.py (paths.bytetrack_repo).

Runtime values come from KDS_* environment variables set by train/run.py so the
same Exp file is used for baseline and treatment.
"""

import functools
import os
from pathlib import Path
import random
import sys

import cv2
import torch
import torch.distributed as dist

# ByteTrack runs tools/train.py as __main__ and its get_exp_by_file imports this
# file with only *this* directory added to sys.path, so a bare ``import train``
# resolves to ByteTrack's own tools/train.py — a module, not our package, which
# fails with "'train' is not a package". Put the repo root first for this one
# import and take it straight back off, so the rest of the run cannot have a
# top-level ByteTrack import shadowed by a same-named directory of ours.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_ADDED_REPO_ROOT = _REPO_ROOT not in sys.path
if _ADDED_REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)
try:
    from train.run import PHOTOMETRIC_STEPS
finally:
    if _ADDED_REPO_ROOT:
        sys.path.remove(_REPO_ROOT)

from yolox.data import TrainTransform, data_augment, get_yolox_datadir
from yolox.exp import Exp as BaseExp


def _distort_subset(image, steps):
    """ByteTrack's ``_distort`` with each of its four steps individually gated.

    A transcription of ``data_augment._distort`` (same order, same ranges, same
    50% gate per step) because the original bundles all four into one function
    with no way to drop one. The BGR->HSV->BGR round trip is lossy and the
    original always pays it, so it is kept whenever an HSV step is selected and
    skipped only when neither is — never adding a conversion the original would
    not have done.
    """
    def _convert(target, alpha=1, beta=0):
        tmp = target.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        target[:] = tmp

    image = image.copy()

    if "brightness" in steps and random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if "contrast" in steps and random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    if not steps & {"hue", "saturation"}:
        return image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if "hue" in steps and random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if "saturation" in steps and random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


class SelectiveTransform(TrainTransform):
    """``TrainTransform`` with the flip and each photometric step switchable.

    ByteTrack's ``TrainTransform`` calls ``_distort`` and ``_mirror``
    unconditionally and never reads its own ``p``, so passing ``p=0`` does not
    turn the flip off. Swapping the two module functions for the duration of the
    call leaves every other step — the degenerate-box filter, the empty-result
    fallback, ``preproc``, the label padding — byte-identical to the stock path,
    which a reimplementation of ``__call__`` could not guarantee.

    One call at a time per process, which is how DataLoader workers use it.
    """

    def __init__(self, *args, flip=True, photometric=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.flip = bool(flip)
        self.photometric = frozenset(photometric)
        unknown = self.photometric - set(PHOTOMETRIC_STEPS)
        if unknown:
            raise ValueError(
                f"unknown photometric steps {sorted(unknown)}; "
                f"expected a subset of {list(PHOTOMETRIC_STEPS)}"
            )

    def __call__(self, image, targets, input_dim):
        distort, mirror = data_augment._distort, data_augment._mirror
        steps = self.photometric
        data_augment._distort = (
            (lambda img: _distort_subset(img, steps)) if steps else (lambda img: img)
        )
        if not self.flip:
            data_augment._mirror = lambda img, boxes: (img, boxes)
        try:
            return super().__call__(image, targets, input_dim)
        finally:
            data_augment._distort = distort
            data_augment._mirror = mirror


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = int(os.environ.get("KDS_NUM_CLASSES", "4"))
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.environ.get("KDS_EXPERIMENT_NAME", "yolox_x_kitti")
        self.data_dir = os.environ.get("KDS_YOLOX_DATA_DIR", get_yolox_datadir())
        self.train_ann = os.environ.get("KDS_YOLOX_TRAIN_ANN", "baseline_train.json")
        self.val_ann = os.environ.get("KDS_YOLOX_VAL_ANN", "eval.json")
        input_size = os.environ.get("KDS_INPUT_SIZE", "800,1440")
        self.input_size = tuple(int(value) for value in input_size.split(","))
        self.test_size = self.input_size
        self.random_size = (18, 32)
        self.max_epoch = int(os.environ.get("KDS_MAX_EPOCH", "60"))
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.01
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        # Keep the held-out KITTI validation split untouched until final TrackEval.
        self.disable_train_eval = True
        # Enter no-Mosaic/MixUp mode at human-readable epoch 51 for a 60-epoch run.
        self.strict_no_aug_boundary = True
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.seed = int(os.environ.get("KDS_SEED", "0"))
        self.output_dir = os.environ.get("KDS_YOLOX_OUTPUT_DIR", "./YOLOX_outputs")

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            MOTDataset,
            YoloBatchSampler,
        )

        # Augmentation toggles (KDS_* env from train/run.py). mosaic_mixup =
        # mosaic + mixup + the affine that YOLOX bundles inside the mosaic
        # pipeline. Flip and the photometric steps are hardcoded inside
        # TrainTransform, so restricting them means swapping the class — see
        # SelectiveTransform. The multi-scale resize (self.random_size, driven by
        # the trainer) is unaffected by any of these toggles.
        flip = os.environ.get("KDS_FLIP", "1") != "0"
        photometric = frozenset(
            step
            for step in os.environ.get("KDS_PHOTOMETRIC", ",".join(PHOTOMETRIC_STEPS)).split(",")
            if step
        )
        mosaic_mixup = os.environ.get("KDS_MOSAIC_MIXUP", "1") != "0"
        # Stock TrainTransform when nothing is restricted, so the arms that keep
        # every augmentation are not routed through the swapping path at all.
        transform = (
            TrainTransform
            if flip and photometric == frozenset(PHOTOMETRIC_STEPS)
            else functools.partial(SelectiveTransform, flip=flip, photometric=photometric)
        )
        use_mosaic = mosaic_mixup and not no_aug

        dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=transform(
                p=0.5,
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
        dataset = MosaicDetection(
            dataset,
            mosaic=use_mosaic,
            img_size=self.input_size,
            preproc=transform(
                p=0.5,
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup and mosaic_mixup,
        )
        self.dataset = dataset
        if is_distributed:
            batch_size //= dist.get_world_size()
        sampler = InfiniteSampler(len(dataset), seed=self.seed)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=use_mosaic,
        )
        return DataLoader(
            dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            if is_distributed
            else torch.utils.data.SequentialSampler(dataset)
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
