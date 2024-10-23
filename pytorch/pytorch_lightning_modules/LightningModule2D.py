import os
import time

import torch
from pytorch.utils.heatmaps import generate_heatmap_target, exact_argmax2d
from pytorch.utils.perturbation import simulate_occlusion_fingers
import pytorch_lightning as pl
from pytorch.models.unet_models import UNet2D
from pytorch.models.scn_models import SCN2D
from pytorch.models.GAFFA import GAFFA
import pandas as pd
from torchvision.transforms import RandomErasing
from medical_data_augment_tool.utils.landmark.landmark_statistics import LandmarkStatistics
from medical_data_augment_tool.utils.landmark.heatmap_test import HeatmapTest
from medical_data_augment_tool.utils.landmark.heatmap_image_generator import HeatmapImageGenerator
from medical_data_augment_tool.utils.io.image import write_np
from medical_data_augment_tool.xray_landmark_heatmap_augmentation import spatial_transformation
from pytorch.utils.landmarks import create_landmarks, compute_metrics, compute_GAFFA_metrics
from pytorch.utils.visualization import save_images_with_landmarks
import lbp_postprocessing as lbp
from os.path import exists
from os.path import join
from os import mkdir
import numpy as np


def initialize_loc_model(config_dic):
    model_classes = {"UNet2D": UNet2D, "SCN2D": SCN2D}
    model_class = model_classes.get(config_dic["model_name"])
    assert model_class, "Model name not recognized"
    fixed_upsampling = True
    if "fixed_upsampling" in config_dic:
        fixed_upsampling = config_dic["fixed_upsampling"]
    return model_class(in_channels=1, out_channels=config_dic["num_landmarks"], fixed_upsampling=fixed_upsampling)


def initialize_loss_function(config_dic):
    loss_functions = {"MSE": torch.nn.functional.mse_loss}
    loss_function = loss_functions.get(config_dic["loss_function"])
    assert loss_function, "Loss function not recognized"
    return loss_function


class LightningModule2D(pl.LightningModule):
    def __init__(self, config_dic: {}, cv: int):
        super().__init__()
        self.max_epoch_nr = config_dic["max_epochs"]
        self.lr = config_dic["initial_learning_rate"]
        self.dim = config_dic["dim"]
        self.num_landmarks = config_dic["num_landmarks"]
        self.test_num_landmarks = config_dic["num_landmarks"] * config_dic["max_test_samples"]
        self.loc_model_name = config_dic["model_name"]
        self.loc_model = initialize_loc_model(config_dic)
        self.heatmap_scale = config_dic["heatmap_scale"]
        self.loss_function_name = config_dic["loss_function"]
        self.loss_function = initialize_loss_function(config_dic)
        self.blobs_distr = config_dic["blobs_distr"]
        self.sigmas = torch.full([self.num_landmarks], config_dic["heatmap_blob_sigma"], device="cuda",
                                 dtype=torch.float32)

        self.save_hyperparameters()
        self.heatmap_test = HeatmapTest(channel_axis=0, invert_transformation=False)
        self.debug_save = config_dic["debug_save"]
        self.debug_save_dir = join("debug_val_predictions_batch")
        self.output_heatmap_size = config_dic["output_size_model"]
        self.heatmap_generator = HeatmapImageGenerator(self.output_heatmap_size,
                                                       config_dic["heatmap_blob_sigma"], 1.0,
                                                       normalize_center=True)
        self.landmark_statistics = LandmarkStatistics()
        self.GAFFA_landmark_statistics = LandmarkStatistics()
        self.lbp_postprocessing = config_dic["lbp_postprocessing"]
        self.GAFFA = config_dic["GAFFA"]
        if self.GAFFA:
            heatmap_size = config_dic["output_size_model"]
            self.GAFFA = GAFFA(num_landmarks=config_dic["num_landmarks"], heatmap_size=heatmap_size,
                               cur_cv=cv, pretrained="load_checkpoint" in config_dic,
                               direct_train_cond= not config_dic["trainable_cond_heatmaps_paper"])
        self.GAFFA_loss_calibrated = False
        self.GAFFA_loss_factor = 0.0
        self.test_outputs = []
        self.val_outputs = []

        self.epoch_nr = 0

        self.occlusion_test = config_dic["occlusion_test"]
        self.occlusion_train = config_dic["occlusion_train"]
        self.occlusion_test_radius = config_dic["occlusion_test_radius"]

        if self.debug_save and not exists(self.debug_save_dir):
            mkdir(self.debug_save_dir)

        # expects learned graph model with correct num_train_samples
        if self.lbp_postprocessing:
            file1 = open('configs/MRF_connectivity/xray_hand_edges.txt', 'r')
            Lines = file1.readlines()
            edges_indices = set()
            column_names = []
            for line in Lines:
                column_names.append(line.rstrip())
                a, b = str.split(line, " ")
                edges_indices.add((int(a) - 1, int(b) - 1))
            self.edges_indices = sorted(edges_indices)
            path = os.path.join("graph_distr", str(cv), str(config_dic["num_train_samples"]), "t_distribs_norm.csv")
            self.distance_distr = np.array(pd.read_csv(path, header=None))

    def forward(self, x):
        return self.loc_model(x)

    def training_step(self, batch, batch_idx):
        img, _, target_landmarks_heatmap_space, _, _ = batch

        # imshow_batch(img.cpu(), "input")

        if batch_idx == 0:
            self.epoch_nr += 1

        p = 0.0
        if self.occlusion_train:
            p = 1.0

        self.log("occlusion_%", p)
        img = RandomErasing(p=p, scale=(0.01, 0.15), value='random')(img)
        # save_image(img, "current_occlusion.png")

        if self.loc_model_name == "SCN2D":
            target_heatmap_hat, _, _ = self.loc_model(img)
        else:
            target_heatmap_hat = self.loc_model(img)

        GAFFA_loss = 0.0
        if self.GAFFA:
            GAFFA_coords_hat = self.GAFFA(target_heatmap_hat)
            # only valid landmarks (flag set by medical augment tool) get used in loss computation
            GAFFA_loss = torch.nn.functional.mse_loss(
                GAFFA_coords_hat[target_landmarks_heatmap_space[:, :, 0] > 0.0][:, 1:],
                target_landmarks_heatmap_space[target_landmarks_heatmap_space[:, :, 0] > 0.0][:, 1:])
            self.log("t", self.GAFFA.temperature)

        target_heatmap = generate_heatmap_target(list(reversed(self.output_heatmap_size)),
                                                 target_landmarks_heatmap_space,
                                                 self.sigmas, scale=self.heatmap_scale,
                                                 distribution=self.blobs_distr)

        # only valid landmarks (flag set by medical augment tool) get used in loss computation
        model_loss = self.loss_function(target_heatmap_hat[target_landmarks_heatmap_space[:, :, 0] > 0.0],
                                            target_heatmap[target_landmarks_heatmap_space[:, :, 0] > 0.0])

        if self.GAFFA and self.GAFFA_loss_calibrated is False:
            self.GAFFA_loss_factor = (model_loss.item() / GAFFA_loss.item())
            self.GAFFA_loss_calibrated = True
        GAFFA_loss *= self.GAFFA_loss_factor

        train_loss = model_loss + GAFFA_loss
        self.log("GAFFA_loss", GAFFA_loss)
        self.log("loc_model_loss", model_loss)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        img, target_landmarks, target_landmarks_heatmap_space, img_ids, img_sizes = batch
        if self.occlusion_test:
            img = simulate_occlusion_fingers(img, target_landmarks_heatmap_space, self.occlusion_test_radius)

        target_heatmap = generate_heatmap_target(list(reversed(self.output_heatmap_size)),
                                                 target_landmarks_heatmap_space,
                                                 self.sigmas, scale=self.heatmap_scale,
                                                 distribution=self.blobs_distr).to(torch.float32)
        start = time.time()
        target_heatmap_hat, local_component, spatial_component = None, None, None
        if self.loc_model_name == "SCN2D":
            target_heatmap_hat, local_component, spatial_component = self.loc_model(img)
        else:
            target_heatmap_hat = self.loc_model(img)
        end = time.time()
        inference_time_batch = end - start

        val_loss = self.loss_function(target_heatmap_hat[target_landmarks_heatmap_space[:, :, 0] > 0.0],
                                          target_heatmap[target_landmarks_heatmap_space[:, :, 0] > 0.0])

        if self.debug_save:
            loc_model_coords_hat = []
            for heatmap in target_heatmap_hat:
                loc_model_coords_hat.append(exact_argmax2d(heatmap))
            save_images_with_landmarks(img, target_landmarks_heatmap_space, img_ids, loc_model_coords_hat, "loc")

        # Use GAFFA to predict coordinates
        if self.GAFFA:
            GAFFA_coords_hat = self.GAFFA(target_heatmap_hat, img_ids, self.debug_save)
            if self.debug_save:
                save_images_with_landmarks(img, target_landmarks_heatmap_space, img_ids, GAFFA_coords_hat, "GAFFA")
            # Compare GAFFA predicted coordinates with target coordinates
            GAFFA_val_loss = torch.nn.functional.mse_loss(
                GAFFA_coords_hat[target_landmarks_heatmap_space[:, :, 0] > 0.0][:, 1:],
                target_landmarks_heatmap_space[target_landmarks_heatmap_space[:, :, 0] > 0.0][:, 1:])
            img_id = 0
            for landmark_prediction, landmark_target in zip(GAFFA_coords_hat.cpu(),
                                                            target_landmarks_heatmap_space.cpu()):
                landmark_prediction_valid = create_landmarks(landmark_prediction)
                landmark_target = create_landmarks(landmark_target)
                self.GAFFA_landmark_statistics.add_landmarks(img_ids[img_id], landmark_prediction_valid,
                                                             landmark_target,
                                                             normalization_factor=50,
                                                             normalization_indizes=[1, 5])
                test = LandmarkStatistics()
                test.add_landmarks(img_ids[img_id], landmark_prediction_valid, landmark_target,
                                   normalization_factor=50, normalization_indizes=[1, 5])
                outliers = test.get_num_outliers([10.0])
                if self.debug_save and outliers[0] > 0:
                    print("GAFFA " + img_ids[img_id] + " number of 10mm spatial outliers: " + str(outliers[0]))
                img_id += 1
            # Logging the GAFFA validation loss
            self.log("GAFFA_validation_loss", GAFFA_val_loss)

        if self.loc_model_name == "SCN2D" and self.debug_save:
            img_id = 0
            for cur_local_heatmap, cur_spatial_heatmap in zip(local_component.cpu(), spatial_component.cpu()):
                debug_path_local_heatmap = join(self.debug_save_dir, img_ids[img_id] + "_local_heatmap.mha")
                debug_path_local_heatmap_2D = join(self.debug_save_dir, img_ids[img_id] + "_local_heatmap_2D.mha")
                debug_path_spatial_heatmap = join(self.debug_save_dir, img_ids[img_id] + "_spatial_heatmap.mha")
                debug_path_spatial_heatmap_2D = join(self.debug_save_dir,
                                                     img_ids[img_id] + "_spatial_heatmap_2D.mha")
                write_np(cur_local_heatmap, debug_path_local_heatmap)
                write_np(cur_local_heatmap.sum(axis=0), debug_path_local_heatmap_2D)
                write_np(cur_spatial_heatmap, debug_path_spatial_heatmap)
                write_np(cur_spatial_heatmap.sum(axis=0), debug_path_spatial_heatmap_2D)
                img_id += 1

        landmark_statistics = self.landmark_statistics
        landmarks = {}
        img_id = 0
        for cur_img, cur_target_heatmap, cur_predicted_heatmap, cur_target_points, cur_target_landmarks_heatmap_space, \
                cur_img_size1, cur_img_size2 \
                in zip(img.cpu(), target_heatmap.cpu(), target_heatmap_hat.cpu(), target_landmarks.cpu(),
                       target_landmarks_heatmap_space.cpu(), img_sizes[0].cpu(), img_sizes[1].cpu()):
            org_predicted_heatmap = cur_predicted_heatmap
            if self.lbp_postprocessing:
                start = time.time()
                cur_predicted_heatmap = lbp.apply_lbp_ratio_pdf(cur_predicted_heatmap, self.distance_distr,
                                                                self.edges_indices,
                                                                self.heatmap_generator)
                end = time.time()
                inference_time_batch = inference_time_batch + end - start
            predicted_landmarks = self.heatmap_test.get_landmarks(cur_predicted_heatmap,
                                                                  transformation=spatial_transformation(
                                                                      self.output_heatmap_size, self.dim)
                                                                  .get(input_size=(cur_img_size1.item(),
                                                                                   cur_img_size2.item()),
                                                                       input_spacing=(1.0, 1.0)))
            cur_target_landmarks = create_landmarks(cur_target_points)

            landmarks[img_id] = predicted_landmarks
            test = LandmarkStatistics()
            test.add_landmarks(img_ids[img_id], predicted_landmarks, cur_target_landmarks,
                               normalization_factor=50, normalization_indizes=[1, 5])
            outliers = test.get_num_outliers([2.0, 4.0, 10.0])

            if self.debug_save and int(outliers[2]) > 0:
                print(img_ids[img_id] + " number of 10mm outliers: " + str(outliers[2]))
                debug_path_refined_heatmap = join(self.debug_save_dir, img_ids[img_id] + "_heatmap_refined.mha")
                debug_path_refined_heatmap_2D = join(self.debug_save_dir, img_ids[img_id] +
                                                     "_heatmap_refined_2D.mha")
                debug_path_img = join(self.debug_save_dir, img_ids[img_id] + "_input.mha")
                debug_path_heatmap = join(self.debug_save_dir, img_ids[img_id] + "_heatmap.mha")
                debug_path_target_heatmap = join(self.debug_save_dir, img_ids[img_id] + "_target_heatmap.mha")
                debug_path_target_heatmap_2D = join(self.debug_save_dir, img_ids[img_id] + "_target_heatmap_2D.mha")

                predicted_landmarks_downsized = self.heatmap_test.get_landmarks(cur_predicted_heatmap)
                refined_heatmap = self.heatmap_generator.generate_heatmaps(predicted_landmarks_downsized, 0)
                refined_heatmap_2D = refined_heatmap.sum(axis=0)
                write_np(refined_heatmap, debug_path_refined_heatmap)
                write_np(refined_heatmap_2D, debug_path_refined_heatmap_2D)
                stacked_img = np.squeeze(np.stack((cur_img,) * self.num_landmarks, axis=1))
                write_np(stacked_img, debug_path_img)
                write_np(org_predicted_heatmap, debug_path_heatmap)
                cur_target_heatmap_2D = cur_target_heatmap.sum(axis=0)
                write_np(cur_target_heatmap, debug_path_target_heatmap)
                write_np(cur_target_heatmap_2D, debug_path_target_heatmap_2D)

            landmark_statistics.add_landmarks(img_ids[img_id], predicted_landmarks, cur_target_landmarks,
                                              normalization_factor=50, normalization_indizes=[1, 5])
            img_id += 1

        self.val_outputs.append([val_loss.cpu(), inference_time_batch])
        return [val_loss.cpu(), inference_time_batch]

    def on_validation_epoch_end(self):
        npy_outputs = np.array(self.val_outputs)
        metrics = compute_metrics(self.landmark_statistics, self.test_num_landmarks)
        GAFFA_metrics = compute_GAFFA_metrics(self.GAFFA_landmark_statistics,
                                              self.test_num_landmarks)
        metrics["val_loss"] = npy_outputs[:, 0].mean()
        metrics["inference_time_batch_sec"] = npy_outputs[:, 1].mean()
        self.landmark_statistics = LandmarkStatistics()
        self.GAFFA_landmark_statistics = LandmarkStatistics()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(GAFFA_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.test_outputs.append(self.validation_step(batch, batch_idx))

    def on_test_epoch_end(self):
        npy_outputs = np.array(self.test_outputs)
        metrics = compute_metrics(self.landmark_statistics, self.test_num_landmarks)
        GAFFA_metrics = compute_GAFFA_metrics(self.GAFFA_landmark_statistics,
                                              self.test_num_landmarks)
        metrics["test_loss"] = npy_outputs[:, 0].mean()
        metrics["inference_time_batch_sec"] = npy_outputs[:, 1].mean()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(GAFFA_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return {
            "optimizer": optimizer
        }
