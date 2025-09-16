from careamics import CAREamist
from careamics.config import (
    create_n2v_configuration,
)
from careamics.lightning.train_data_module import TrainDataModule
import zarr
import numpy as np
from .normalizations import minmax
from torch import load

def n2v_denoise():

    img = zarr.open("/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0/0")
    img = np.squeeze(img)
    img = minmax(img[0])

    # path_to_zarr = "/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0/0"

    # Create a configuration using the helper function
    training_config = create_n2v_configuration(
        experiment_name="test2_n2v_exp", 
        data_type="array",
        axes="YX",
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

    careamist = CAREamist(source=training_config)
    # train_datamodule = TrainDataModule(
    #     data_config=training_config.data_config,
    #     train_data=path_to_zarr,
    #     read_source_func=zarr.open,
    #     use_in_memory=False
    # )

    careamist.train(train_source=img)
    prediction = careamist.predict(
        source=img,
        dataloader_params={"num_workers": 7, "persistent_workers": True}
        )
    print(type(prediction), len(prediction), prediction[0].shape)
    
    np.save("/Users/chrisviets/Desktop/denoised.npy", prediction[0].squeeze())

    return

def predict_from_ckpt(path_to_ckpt, path_to_img):

    ckpt = load(path_to_ckpt, map_location="cpu")
    print(f"Checkpoint from epoch: {ckpt['epoch']}")

    img = zarr.open(path_to_img)
    img = np.squeeze(img)
    img = minmax(img[0])
    pretrained_careamist = CAREamist(source=path_to_ckpt)
    new_preds = pretrained_careamist.predict(source=img, tile_size=(256, 256))[0]
    np.save("/Users/chrisviets/Desktop/denoised.npy", new_preds.squeeze())

def main_denoise():

    n2v_denoise()

    # path_to_ckpt = "/Users/chrisviets/Documents/hcsegment/checkpoints/TEST_n2v_exp-v2-1.ckpt"
    # path_to_img = "/Volumes/Chris2/Exp001E/HCS_zarr.zarr/A/10/0/0"

    # predict_from_ckpt(path_to_ckpt, path_to_img)

