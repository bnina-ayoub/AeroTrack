import os
from aerotrack.exp.yolox_base import Exp as YoloXBaseExperiment
import torch.distributed
import torch
import torch.nn as nn
class UAVSwarmBaseExperiment(YoloXBaseExperiment):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.test_size = (640, 640)
        self.input_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.7
        self.data_dir = "dataset/UAVSwarm"
        self.train_ann = "train.json"
        self.val_ann = "test.json"

    def configure_batch_normalization(self, module):
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from aerotrack.data import MOTDataset, TrainTransform, InfiniteSampler, YoloBatchSampler, DataLoader, MosaicDetection
        from aerotrack.utils import wait_for_the_master
        
        with wait_for_the_master():
            base_dataset = MOTDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="train",
                img_size=self.input_size,
                preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            )
        augmented_dataset = MosaicDetection(
            base_dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        sampler = InfiniteSampler(len(augmented_dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        return DataLoader(augmented_dataset, num_workers=self.data_num_workers, pin_memory=True, batch_sampler=batch_sampler)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from aerotrack.data import MOTDataset, ValTransform
        val_dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name="test",
            preproc=ValTransform(),
        )
        sampler = torch.utils.data.SequentialSampler(val_dataset)
        return torch.utils.data.DataLoader(val_dataset, num_workers=self.data_num_workers, pin_memory=True, sampler=sampler, batch_size=batch_size)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from aerotrack.evaluators import COCOEvaluator
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )