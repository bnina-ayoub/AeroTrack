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
        self.val_ann = "val_half.json"
        self.data_num_workers = 2
        self.random_size = None
        self.max_epoch = 65
        self.warmup_epochs = 5
        self.warmup_lr = 0.0
        self.basic_lr_per_img = 0.01 / 8.0
        self.scheduler = "warmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.0
        self.ema = False
        self.eval_interval = 1
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.tail_lr_factor = 0.03

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

    def get_optimizer(self, batch_size):
        import torch

        if "optimizer" not in self.__dict__:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.warmup_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        return self.optimizer

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from aerotrack.data import MOTDataset, ValTransform
        val_dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name="train",
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