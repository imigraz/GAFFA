import os
import random
import torch
from torch.utils.data import DataLoader
from os.path import exists
from os.path import join
from os import mkdir
from medical_data_augment_tool.xray_landmark_heatmap_augmentation import XrayDataAugmentationLegacy
import pytorch_lightning as pl


class XrayHandDataModule(pl.LightningDataModule):
    def __init__(self, config_dic: {}, cur_cv_nr: int):
        super().__init__()
        self.batch_size = config_dic["batch_size"]
        self.train_dataset = XrayHandDataset(config_dic, True, cur_cv_nr)
        self.test_dataset = XrayHandDataset(config_dic, False, cur_cv_nr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class XrayHandDataset(torch.utils.data.Dataset):
    def __init__(self, config_dic: {}, train: bool, cur_cv_nr) -> None:
        super().__init__()

        data_dir = config_dic["dataset_path"]
        self.data_dir_images = join(data_dir, "images")
        self.data_dir_setup = join(data_dir, "setup")
        self.num_landmarks = config_dic["num_landmarks"]
        self.cv_set = cur_cv_nr
        self.train = train
        if train:
            self.num_samples = config_dic["num_train_samples"]
        else:
            self.num_samples = config_dic["max_test_samples"]
        self.debug_save = config_dic["debug_save"]
        self.heatmap_scale = config_dic["heatmap_scale"]

        which_data = "train" if train else "test"
        self.debug_save_dir = "debug_input" + "_" + which_data
        if self.debug_save and not exists(self.debug_save_dir):
            mkdir(self.debug_save_dir)

        # get train/test image paths
        which_data = "train.txt" if train else "test.txt"
        which_cv = "cv"

        # for original reduced cv split:
        # which_cv = "cv_reduced" if cv_reduced else "cv"

        cv_set_txt = join(self.data_dir_setup, which_cv, "set" + str(self.cv_set), which_data)

        self.augmentor = XrayDataAugmentationLegacy(config_dic, self.data_dir_images,
                                                    join(self.data_dir_setup, "all.csv"),
                                                    cv_set_txt, train)

        with open(cv_set_txt) as file:
            self.xray_files = [join(line.rstrip() + config_dic["input_image_ext"]) for line in file]

        if train and self.num_samples < len(self.xray_files):
            samples_file_path = join(self.data_dir_setup, f"random_samples_cv{self.cv_set}_n{self.num_samples }.txt")
            if not exists(samples_file_path) or os.stat(samples_file_path).st_size == 0:
                with open(cv_set_txt) as file:
                    all_files = [join(line.rstrip() + config_dic["input_image_ext"]) for line in file]
                self.xray_files = random.sample(all_files, min(self.num_samples, len(all_files)))
                with open(samples_file_path, 'w') as f:
                    for item in self.xray_files:
                        f.write("%s\n" % item)
            else:
                with open(samples_file_path, 'r') as f:
                    self.xray_files = [line.strip() for line in f]
        print(which_data + " files: " + str(len(self.xray_files)))

    def __len__(self):
        return len(self.xray_files)

    def __getitem__(self, index):
        xray_file_name_no_ext = self.xray_files[index].split('.')[0]
        xray_image, xray_landmark_2Dpoints, xray_landmark_2Dpoints_target, img_size = \
            self.augmentor.get_data(xray_file_name_no_ext)

        return xray_image, xray_landmark_2Dpoints, xray_landmark_2Dpoints_target, xray_file_name_no_ext, img_size
