from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from datamodule import MVTecDataModule
from network.xxxxx import xxxxx


def train(hparams):
    dict_hparams = vars(hparams)

    modelXXX = xxxxx(**dict_hparams)

    data_module = MVTecDataModule(
        json_path=hparams.train_json_path,
        train_batch_size=hparams.train_batch_size,
        validation_batch_size=hparams.validation_batch_size,
        input_image_size=hparams.input_image_size
    )

    logger = TestTubeLogger(hparams.log_dir, name=hparams.model_name)
    checkpoint_callback = ModelCheckpoint(
        monitor=hparams.monitoring_loss,
        mode="min",
        dirpath=f"{hparams.output_dir}/model",
        filename="{epoch}-{val_loss:.2f}",
        save_last=True
    )
    trainer = Trainer.from_argparse_args(hparams, logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(modelXXX, data_module)


if __name__ == "__main__":

    parser = ArgumentParser()
    # general parameter
    parser.add_argument("--model_name", type=str, default="xxx")
    parser.add_argument("--train_json_path", type=str, default="./data.json")
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--input_image_size", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--validation_batch_size", type=int, default=1)
    parser.add_argument("--monitoring_loss", type=str, default="val_reconstruct_loss")
    parser.add_argument("--discription", type=str, default="discription")

    parser = xxxxx.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    Path(hparams.output_dir).mkdir(parents=True, exist_ok=True)

    train(hparams)
