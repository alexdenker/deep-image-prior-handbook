import torch
import torch.nn as nn
import numpy as np



def get_unet_model(in_ch=1, out_ch=1, scales=5, skip=4,
                   channels=(32, 32, 64, 64, 128, 128), use_sigmoid=True,
                   use_norm=True, activation="relu", padding_mode="circular", 
                   upsample_mode='nearest', initialisation="xavier_uniform"):
    assert activation in ["relu", "silu","leaky_relu"], "Activation function has bo the either ReLU, SiLU LeakyReLU"
    assert (1 <= scales <= 6)
    skip_channels = [skip] * (scales)
    return UNet(in_ch=in_ch, out_ch=out_ch, channels=channels[:scales],
                skip_channels=skip_channels, use_sigmoid=use_sigmoid,
                use_norm=use_norm, activation=activation,padding_mode=padding_mode, 
                upsample_mode=upsample_mode, initialisation=initialisation)



class UNet(nn.Module):

    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True, activation="relu", 
                 padding_mode="circular", upsample_mode='nearest',
                 initialisation="xavier_uniform"):
        super(UNet, self).__init__()
        assert (len(channels) == len(skip_channels))
        assert activation in ["relu", "silu", "leaky_relu"], "Activation function has bo the either ReLU, SiLU LeakyReLU"

        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.activation_type = activation
        self.initialisation = initialisation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm, activation=self.activation,padding_mode=padding_mode )
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm,
                                       activation=self.activation, 
                                       padding_mode=padding_mode))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm,
                                   activation=self.activation, 
                                   padding_mode=padding_mode,
                                   upsample_mode=upsample_mode))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch, 
                             padding_mode=padding_mode)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initialisation == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif self.initialisation == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif self.initialisation == "kaiming_normal":
                    if self.activation_type == "leaky_relu":
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                    else:   
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=self.activation_type)
                elif self.initialisation == "kaiming_uniform":
                    if self.activation_type == "leaky_relu":
                        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                    else:   
                        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=self.activation_type)
                else:
                    raise NotImplementedError(f"Initialisation {self.initialisation} not implemented")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        return torch.sigmoid(self.outc(x)) if self.use_sigmoid else self.outc(x)




class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True, activation=nn.ReLU(),padding_mode="circular"):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad,padding_mode=padding_mode),
                nn.GroupNorm(num_channels=out_ch, num_groups=4),
                activation,
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad, padding_mode=padding_mode),
                nn.GroupNorm(num_channels=out_ch, num_groups=4),
                activation)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad, padding_mode=padding_mode),
                activation,
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad, padding_mode=padding_mode),
                activation)

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True, activation=nn.ReLU(), padding_mode="circular"):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad,padding_mode=padding_mode),
                nn.GroupNorm(num_channels=out_ch, num_groups=2),
                activation)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad, padding_mode=padding_mode),
                activation)



    def forward(self, x):
        x = self.conv(x)
        return x




class UpBlock(nn.Module):

    def __init__(self, in_ch, 
                        out_ch, 
                        skip_ch=4, 
                        kernel_size=3, 
                        use_norm=True,
                        activation=nn.ReLU(), 
                        padding_mode="circular",
                        upsample_mode='nearest'):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_channels=in_ch + skip_ch,  num_groups=1),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad, padding_mode=padding_mode),
                nn.GroupNorm(num_channels=out_ch, num_groups=4),
                activation,
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad, padding_mode=padding_mode),
                nn.GroupNorm(num_channels=out_ch, num_groups=4),
                activation)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad, padding_mode=padding_mode),
                activation,
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad, padding_mode=padding_mode),
                activation)

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1, padding_mode=padding_mode),
                nn.GroupNorm(num_channels=skip_ch, num_groups=1),
                activation)
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1, padding_mode=padding_mode),
                activation)
        if upsample_mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)

        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)



class OutBlock(nn.Module):

    def __init__(self, in_ch, out_ch, padding_mode="circular"):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding_mode=padding_mode)

    def forward(self, x):
        x = self.conv(x)
        return x


    def __len__(self):
        return len(self._modules)