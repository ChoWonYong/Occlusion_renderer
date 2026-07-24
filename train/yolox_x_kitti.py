"""YOLOX-X experiment consumed by /home/wcho/ByteTrack/tools/train.py.

Runtime values come from KDS_* environment variables set by train/run.py so the
same Exp file is used for baseline and treatment.
"""

import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as BaseExp


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
            TrainTransform,
            YoloBatchSampler,
        )

        # Augmentation toggles (KDS_* env from train/run.py). online_jitter =
        # online flip + HSV; mosaic_mixup = mosaic + mixup + the affine that YOLOX
        # bundles inside the mosaic pipeline. TrainTransform.p is the flip prob;
        # note HSV (_distort) is always applied by TrainTransform, so
        # online_jitter=0 only removes the horizontal flip.
        online_jitter = os.environ.get("KDS_ONLINE_JITTER", "1") != "0"
        mosaic_mixup = os.environ.get("KDS_MOSAIC_MIXUP", "1") != "0"
        flip_prob = 0.5 if online_jitter else 0.0
        use_mosaic = mosaic_mixup and not no_aug

        dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=TrainTransform(
                p=flip_prob,
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
        dataset = MosaicDetection(
            dataset,
            mosaic=use_mosaic,
            img_size=self.input_size,
            preproc=TrainTransform(
                p=flip_prob,
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
