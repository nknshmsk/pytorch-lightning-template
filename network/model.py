from argparse import ArgumentParser

from torch import nn
import pytorch_lightning as pl

from util.network_utils import conv_layer, transpose_conv_layer

class ResnetBlock(nn.Module):
    def __init__(
        self,
        channel,
    ):
        super(ResnetBlock, self).__init__()
        sequence = []
        sequence += [conv_layer(channel, channel, padding=1, activation=nn.SiLU())]
        sequence += [conv_layer(channel, channel, padding=1, activation=False)]
        self.res_block = nn.Sequential(*sequence)

    def forward(self, input):
        return self.res_block(input) + input


class xxxxx(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, input_image):
        return 

    def training_step(self, input, batch_idx):
        return

    def validation_step(self, input, batch_idx):
        return

    def validation_epoch_end(self, val_step_output):
        return

    def configure_optimizers(self):
        def lambdalr_func(epoch):
            if epoch < 1000:
                return 0.75**0
            elif epoch < 1500:
                return 0.75**1
            elif epoch < 2000:
                return 0.75**2
            else:
                return 0.75**3

        self.optimizer = torch_optimizer.xxx(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambdalr_func)

        return [self.optimizer], [self.scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)

        return parser
