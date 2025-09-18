from careamics import CAREamist
from careamics.config import (
    create_n2v_configuration,
)
from careamics.lightning import TrainDataModule
import zarr
import numpy as np
from .normalizations import minmax
from torch import load
from glob import glob
import os
import tempfile
import shutil
import tifffile

def n2v_denoise(path_to_tiffs: str):

    path_to_tiffs = os.path.expanduser(path_to_tiffs)

    tiffs = glob(os.path.join(path_to_tiffs), "*.tif")
    tiffs.extend(glob(os.path.join(path_to_tiffs, "*.tiff")))

    example_tiff = tifffile.imread(tiffs[0])
    dimensions = example_tiff.shape
    assert len(dimensions) == 5, "Expected 5-D image (TCZYX)"

    removal_files = glob(os.path.join(os.path.expanduser(path_to_tiffs), "._*.tif*"))
    for file in removal_files:
        os.remove(os.path.join(path_to_tiffs, file))

    axes = "YX"
    if dimensions[2] > 1:
        axes = "Z" + axes
    if dimensions[1] > 1:
        axes = "C" + axes
    if dimensions[0] > 1:
        axes = "S" + axes

    # Create a configuration using the helper function
    training_config = create_n2v_configuration(
        experiment_name="n2v_exp", 
        data_type="tiff",
        axes=axes,
        patch_size=[64, 64],
        batch_size=128,
        num_epochs=3,
        roi_size=11,
        masked_pixel_percentage=0.2,
        train_dataloader_params={'num_workers': 7, "persistent_workers": True},
        val_dataloader_params={'num_workers': 7, 'persistent_workers': True},
        checkpoint_params={
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Created temporary directory: {tmpdirname}")

        careamist = CAREamist(source=training_config, work_dir=tmpdirname)
        train_datamodule = TrainDataModule(
            data_config=training_config.data_config,
            train_data=path_to_tiffs,
            use_in_memory=False
        )

        careamist.train(datamodule=train_datamodule)
        print("Completed training")

        os.mkdir(".tmp")
        shutil.move(os.path.join(tmpdirname, "checkpoints", "n2v_exp.ckpt"),
                    os.path.join(".tmp", "n2v_exp.ckpt"))

        return os.path.join(os.path.expanduser(".tmp"), "n2v_exp.ckpt")
    

def predict_from_ckpt(path_to_ckpt, path_to_img):

    ckpt = load(path_to_ckpt, map_location="cpu")
    print(f"Checkpoint from epoch: {ckpt['epoch']}")

    img = zarr.open(path_to_img)
    img = np.squeeze(img)
    img = minmax(img[0])
    pretrained_careamist = CAREamist(source=path_to_ckpt)
    new_preds = pretrained_careamist.predict(source=img, tile_size=(256, 256))[0]
    np.save("/Users/chrisviets/Desktop/denoised.npy", new_preds.squeeze())

def main_denoise(input):

    n2v_denoise(input)

    # path_to_ckpt = "/Users/chrisviets/Documents/hcsegment/checkpoints/TEST_n2v_exp-v2-1.ckpt"
    # path_to_img = "/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0/0"

    # predict_from_ckpt(path_to_ckpt, path_to_img)

