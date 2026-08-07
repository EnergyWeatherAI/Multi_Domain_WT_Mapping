import torch
import torch.nn as nn

class Residual_TCN_Block(nn.Module):

    def __init__(self, in_ch, feature_maps, kernel, dilation, added_norm = False):
        super().__init__() 
        self.in_ch = in_ch
        self.ch_out = feature_maps

        self.match_dim = self.in_ch != self.ch_out 
        if self.match_dim:
            self.dim_conv = nn.Conv1d(in_ch, feature_maps, kernel_size=1, bias=False)
            

        bias = not added_norm 
        self.conv1 = nn.Conv1d(in_ch, feature_maps, kernel, padding="same", dilation=dilation, bias=bias, padding_mode="reflect")
        self.conv2 = nn.Conv1d(feature_maps, feature_maps, kernel, padding="same", dilation=dilation, bias=bias, padding_mode="reflect")
        self.act = nn.Mish() 
        self.norm = nn.GroupNorm(1, feature_maps, affine=False) if added_norm else None 


    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        if self.norm is not None:
            h = self.norm(h)
        identity = self.dim_conv(x) if self.match_dim else x 
        return h + identity

class GeneratorTCN(nn.Module):

    def __init__(self, n_feat = 11, n_cls=5):
        super().__init__()
        self.n_cls = n_cls
        fmaps = 128
        norm = True
        self.block1 = Residual_TCN_Block(in_ch=n_feat+n_cls, feature_maps = fmaps, kernel=3, dilation=1, added_norm=False)
        self.block2 = Residual_TCN_Block(in_ch=fmaps, feature_maps = fmaps, kernel=3, dilation=2, added_norm=norm)        
        self.block3 = Residual_TCN_Block(in_ch=fmaps, feature_maps = fmaps, kernel=3, dilation=4, added_norm=norm)
        
        self.block4 = Residual_TCN_Block(in_ch=fmaps, feature_maps = fmaps, kernel=3, dilation=8, added_norm=norm)
        self.block5 = Residual_TCN_Block(in_ch=fmaps, feature_maps = fmaps, kernel=3, dilation=16, added_norm=norm)
        self.block6 = Residual_TCN_Block(in_ch=fmaps, feature_maps = fmaps, kernel=3, dilation=32, added_norm=False)

        # ---- OUTPUT CONV ------
        self.conv_out = torch.nn.Sequential(
                        nn.Conv1d(in_channels=fmaps, out_channels = n_feat, bias=True, kernel_size=1, stride=1))

    
    def forward(self, x, destination_y):
        n_btch, n_ch, len_x = x.size() # 64 x 11 x 72
        c_d = destination_y.unsqueeze(2).expand((n_btch, self.n_cls, len_x)) 
        x_and_l = torch.cat([x, c_d], dim=1)
        x = self.block1(x_and_l)
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_out(x)
        return x 


class Discriminator(nn.Module):
    def __init__(self, n_feat = 11, n_cls = 6):
        super().__init__()

        # ----- ENCODE ------
        # block 1
        self.block1 = nn.Sequential(
                                        nn.Conv1d(n_feat, 256, bias=False, kernel_size=5, stride=2, padding=2, padding_mode="reflect"),
                                        nn.Mish(),
                                        nn.GroupNorm(1, 256)
        ) # -> 36


        self.block2 = nn.Sequential(
                                        nn.Conv1d(256, 256, bias=False, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                        nn.Mish(),
                                        nn.GroupNorm(1, 256)
        ) # -> 18



        self.block3 = nn.Sequential(
                                        nn.Conv1d(256, 512, bias=False, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                        nn.Mish(),
                                        nn.GroupNorm(1, 512)
        ) # -> 9

        self.out_D = nn.Sequential(nn.Flatten(), nn.Linear(9 * 512, 1))
        self.out_cls = nn.Sequential(nn.Flatten(), nn.Linear(9 * 512, n_cls))


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        d = self.out_D(x)
        c = self.out_cls(x)
        return d, c