import math

from datasets.xray_hand_dataset import XrayHandDataModule
from pytorch.models.GAFFA import learn_conditional_distr
from pytorch.pytorch_lightning_modules import LightningModule2D
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torchinfo import summary
import json
from lbp_postprocessing import train_graph
import argparse
import os


def initialize_wandb_logger(config_dic: {}):
    # start a new wandb run to track this script
    runname = None
    if "wandb_runname" in config_dic:
        runname = config_dic["wandb_runname"]
    wandb_logger = WandbLogger(name=runname, project=config_dic["wandb_projectname"], log_model=True)
    return wandb_logger


def initialize_datamodule(config_dic: {}, cur_cv_nr):
    init_datamodule = None
    dataset_name = config_dic["dataset_name"]
    if dataset_name == "xray_hand":
        init_datamodule = XrayHandDataModule(config_dic, cur_cv_nr)
    assert init_datamodule
    return init_datamodule


def initialize_lightning_module(config_dic: {}, cur_cv_nr):
    init_model = None
    dim = config_dic["dim"]
    if dim == 2:
        init_model = LightningModule2D.LightningModule2D(config_dic, cur_cv_nr)
    assert init_model
    return init_model


if __name__ == "__main__":
    # name of experiment config JSON file to be used, read from command line
    parser = argparse.ArgumentParser("Anatomical Landmark Localization Framework")
    parser.add_argument("config_file", help="The desired config file name (with ext) to be used for the experiment. "
                                            "We assume that it is located in configs/experimental_scripts", type=str)
    args = parser.parse_args()

    config_path = os.path.join('configs', 'experiments', args.config_file)
    f = open(config_path)
    config_dic = json.load(f)
    f.close()

    # set tensorcore precision
    torch.set_float32_matmul_precision('medium')

    # setup weights and biases logger
    logger = True
    if not config_dic["fast_dev_run_trainer"]:
        logger = initialize_wandb_logger(config_dic)

    config_dic["dataset_path"] = os.path.join(config_dic["datasets_path"], config_dic["dataset_name"])

    # set seeds for numpy, torch and python.random
    if config_dic["deterministic_trainer"]:
        pl.seed_everything(config_dic["seed"], workers=True)

    test_data_all = []
    # setup of dataset and model
    for i, cur_cv_nr in zip(range(len(config_dic["cv_folds"])), config_dic["cv_folds"]):

        data_cur_cv = initialize_datamodule(config_dic, cur_cv_nr)

        if config_dic["train_graph"]:
            train_graph(config_dic, data_cur_cv, cur_cv_nr)
        if config_dic["precompute_cond_heatmaps"]:
            print("Computing conditional distribution heatmaps...")
            learn_conditional_distr(config_dic, data_cur_cv, cur_cv_nr)
            print("Computing conditional distribution heatmaps done!")

        pl_model = initialize_lightning_module(config_dic, cur_cv_nr)
        if not config_dic["fast_dev_run_trainer"]:
            logger.watch(pl_model)
        summary(pl_model, input_size=(1, 1, *config_dic["input_size_model"]))

        if "load_checkpoint" in config_dic and len(config_dic["load_checkpoint"][i]) > 0:
            chk_paths = config_dic["load_checkpoint"]
            cur_ckpt_path = os.path.join("pretrained_models", chk_paths[i])
            pl_model = LightningModule2D.LightningModule2D.load_from_checkpoint(checkpoint_path=cur_ckpt_path, config_dic=config_dic, strict=False)

        # to keep training time similar to full dataset, when using an reduced dataset
        ratio_train_samples = math.ceil(config_dic["max_train_samples"] / config_dic["num_train_samples"])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=ratio_train_samples)
        experiment_dir_name = config_dic["dataset_name"] + "_" + config_dic["model_name"] + "_cv" + str(cur_cv_nr)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config_dic["max_epochs"] * ratio_train_samples,
                             deterministic=config_dic["deterministic_trainer"],
                             fast_dev_run=config_dic["fast_dev_run_trainer"],
                             default_root_dir=experiment_dir_name,
                             check_val_every_n_epoch=ratio_train_samples,
                             log_every_n_steps=config_dic["log_every_n_steps"],
                             logger=logger,
                             enable_model_summary=False,
                             callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
        if config_dic["train_model"]:
            train_loader = data_cur_cv.train_dataloader()
            val_loader = data_cur_cv.test_dataloader()
            trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if config_dic["test"]:
            test_loader = data_cur_cv.test_dataloader()
            test_data = trainer.test(model=pl_model, dataloaders=test_loader)
            test_data_all.append(test_data)

    final_test_dic = {}
    # for each metric, calc avg over all cv to get final scores
    for key in test_data_all[0][0]:
        sum = 0.0
        num_cv = len(test_data_all)
        for test_idx in range(num_cv):
            sum = sum + test_data_all[test_idx][0][key]
        avg = sum / num_cv
        final_test_dic[key] = avg

    if not config_dic["fast_dev_run_trainer"]:
        logger.log_metrics(final_test_dic)

    print("Final test output over all folds:")
    print(json.dumps(final_test_dic, indent=4))
