import torch import nn

def conv_layer(
    channel_in,
    channel_out,
    kernel_size=3,
    stride=1,
    use_bias=False,
    activation=nn.ReLU(True),
    use_norm=True,
    drop_rate=False,
    padding=False,
):

    sequence = []
    sequence = [nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, bias=use_bias, stride=stride, padding_mode="reflect")]
    if use_norm:
        sequence += [nn.BatchNorm2d(channel_out)]
    if activation is not False:
        sequence += [activation]
    if drop_rate is not False:
        sequence += [nn.Dropout(drop_rate)]

    return nn.Sequential(*sequence)


def transpose_conv_layer(
    channel_in,
    channel_out,
    kernel_size=3,
    stride=2,
    use_bias=False,
    activation=nn.ReLU(True),
    use_norm=True,
    drop_rate=False,
    padding=1,
):

    sequence = []
    sequence = [nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, bias=use_bias, stride=stride)]
    if use_norm:
        sequence += [nn.BatchNorm2d(channel_out)]
    if activation is not False:
        sequence += [activation]
    if drop_rate is not False:
        sequence += [nn.Dropout(drop_rate)]

    return nn.Sequential(*sequence)
