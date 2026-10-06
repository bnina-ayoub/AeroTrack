import torch
from aerotrack_proposed import Exp as AeroTrackExperiment


class Exp(AeroTrackExperiment):
    def __init__(self):
        super().__init__()
        self.exp_name = "dut_anti_uav"
        self.data_dir = "dataset/DUT Anti-UAV"
        self.train_ann = "test.json"
        self.val_ann = "test.json"
        self.data_num_workers = 2

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
        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
        )
