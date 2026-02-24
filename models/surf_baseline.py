import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten
import numpy as np
import random
from lib.model_arch import modality_drop, unbalance_modality_drop,modality_drop_dual,unbalance_modality_drop_dual
from src.freq_n import FrequencyLayer

loss_bank_rgb = np.array(0, np.float64)
loss_bank_depth= np.array(0, np.float64)
loss_bank_ir= np.array(0, np.float64)


class SURF_Multi(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4


class SURF_Baseline(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)


        # print(self.drop_mode)

        if self.drop_mode == 'average':
            # print(1)
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            # print(2)
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)



        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4
class SURF_Baseline_dual(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)


        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         # model_resnet18_se_1.dropout,
                                         )
        self.FreqLayer = FrequencyLayer(64, (112, 112, 3))

        self.rgb_cls = nn.Sequential(model_resnet18_se_1.avgpool,
                                     Flatten(1),
                                     nn.Linear(128,2))
        self.depth_cls = nn.Sequential(model_resnet18_se_2.avgpool,
                                     Flatten(1),
                                     nn.Linear(128, 2))

    def forward(self, img_rgb, img_depth):
        rgb_low, rgb_high = self.FreqLayer(img_rgb)
        depth_low, depth_high = self.FreqLayer(img_depth)

        rgb_weight = (rgb_low / (rgb_high + 1e-6)).abs().sum()
        depth_weight = (depth_low / (depth_high + 1e-6)).abs().sum()

        # rgb_weight = 0.8 * ((rgb_low).abs().sum()) + 0.2 * ((rgb_high).abs().sum())
        # depth_weight = 0.8 * ((depth_low).abs().sum()) + 0.2 * ((depth_high).abs().sum())


        r_w, d_w = self.weightFun(rgb_weight, depth_weight)
        # print(r_w,d_w)
        # r_w, d_w = 0,0
        # print(img_depth)
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)

        # print(self.drop_mode)

        # if self.drop_mode == 'average':
        #     # print(1)
        #     x_rgb, x_depth,  p = modality_drop_dual(x_rgb, x_depth,  self.p, self.args)
        # else:
        #     # print(2)
        #     x_rgb, x_depth, p = unbalance_modality_drop(x_rgb, x_depth, self.p, self.args)

        x = torch.cat((x_rgb, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return (x, layer3, layer4),(r_w,d_w)

    def fresh_bank(self,rgb_,depth_):
        global loss_bank_rgb, loss_bank_depth,loss_bank_ir

        if loss_bank_rgb == 0:
            loss_bank_rgb = rgb_
        else:
            loss_bank_rgb = (loss_bank_rgb + rgb_) / 2

        if loss_bank_depth == 0:
            loss_bank_depth = depth_
        else:
            loss_bank_depth = (loss_bank_depth + depth_) / 2


    def weightFun(self, rgb_w, depth_w):
        global loss_bank_rgb, loss_bank_depth,loss_bank_ir
        inpt = [float(rgb_w.cpu().numpy()), float(depth_w.cpu().numpy())]

        alpha = 1.5
        beta = 1
        gama = 0.7
        sigma = 4
        init_weight=0.5

        self.fresh_bank(float(rgb_w.cpu().numpy()), float(depth_w.cpu().numpy()))

        if (loss_bank_rgb == 0 or loss_bank_depth == 0):
            inpt[0] = init_weight
            inpt[1] = init_weight

        else:
            mean = (loss_bank_rgb + loss_bank_depth ) / 2
            aa = inpt[0] / (mean + 1e-6)
            bb = inpt[1] / (mean + 1e-6)

            aa = alpha - beta * (1 / (1 + np.exp(-(aa - gama) * sigma)))
            bb = alpha - beta * (1 / (1 + np.exp(-(bb - gama) * sigma)))

            inpt[0] = aa
            inpt[1] = bb

        return torch.tensor(inpt, dtype=torch.float32)

class SURF_Baseline_Auxi(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 128
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_ir_out = self.auxi_bone(x_ir)
        x_depth_out = self.auxi_bone(x_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)



        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p


class SURF_Baseline_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 128
        self.transformer = nn.Conv2d(128, 128, 1, 1)
        self.transformer_rgb = nn.Conv2d(128, 128, 1, 1)
        self.transformer_depth = nn.Conv2d(128, 128, 1, 1)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_depth_out = self.auxi_bone(x_depth)

        x_rgb_trans = self.transformer(x_rgb)
        x_depth_trans = self.transformer(x_depth)

        x_rgb_depth = (x_rgb_trans + x_depth_trans) / 2
        x_rgb_depth = self.auxi_bone(x_rgb_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)


        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p


class SURF_Baseline_Auxi_Weak_Layer4(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer3_new,
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )



    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)


        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p

class SURF_MMANet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer3_new,
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )

        self.FreqLayer = FrequencyLayer(args.batch_size, (112, 112, 3))

    def forward(self, img_rgb, img_depth, img_ir):
        if self.training:
            rgb_low, rgb_high = self.FreqLayer(img_rgb)
            depth_low, depth_high = self.FreqLayer(img_depth)
            ir_low, ir_high = self.FreqLayer(img_ir)
            rgb_weight = (rgb_low / (rgb_high + 1e-6)).abs().sum()
            depth_weight = (depth_low / (depth_high + 1e-6)).abs().sum()
            ir_weight = (ir_low / (ir_high + 1e-6)).abs().sum()
            r_w, d_w, i_w, = self.weightFun(rgb_weight, depth_weight, ir_weight)
            # print(r_w, d_w, i_w)
        else:
            r_w, d_w, i_w, = 1,1,1
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        if self.training:
            x_rgb_out = self.auxi_bone(x)
            x_rgb_depth = self.auxi_bone(x)
            x_depth_out = self.auxi_bone(x)
        else:
            x_rgb_out = None
            x_rgb_depth = None
            x_depth_out = None
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p,(r_w,d_w,i_w)

    def fresh_bank(self, rgb_,depth_, ir_,alpha=0.5,is_soft_begin=False):

        global loss_bank_rgb, loss_bank_ir,loss_bank_depth

        rgb_ = rgb_.detach().float()
        ir_ = ir_.detach().float()
        depth_ = depth_.detach().float()

        if loss_bank_rgb == 0:
            if is_soft_begin:
                loss_bank_rgb = rgb_ * (1 - alpha)
            else:
                loss_bank_rgb = rgb_
        else:
            loss_bank_rgb = alpha * loss_bank_rgb + (1 - alpha) * rgb_


        if loss_bank_depth == 0:
            if is_soft_begin:
                loss_bank_depth = depth_ * (1 - alpha)
            else:
                loss_bank_depth = depth_
        else:
            loss_bank_depth = alpha * loss_bank_depth + (1 - alpha) * depth_

        if loss_bank_ir == 0:
            if is_soft_begin:
                loss_bank_ir = ir_ * (1 - alpha)
            else:
                loss_bank_ir = ir_
        else:

            loss_bank_ir = alpha * loss_bank_ir + (1 - alpha) * ir_

    def weightFun(self, rgb_w,depth_w, ir_w):
        global loss_bank_rgb, loss_bank_ir,loss_bank_depth

        rgb_w = rgb_w.float()
        ir_w = ir_w.float()
        depth_w = depth_w.float()


        alpha = 1.5  # 1.5
        beta = 1  # 1
        gamma = 1  # 0.7
        sigma = 10  # 6
        init_weight = 1

        self.fresh_bank(rgb_w,depth_w, ir_w,alpha=0.5,is_soft_begin=True)

        out = torch.stack([rgb_w,depth_w, ir_w])  # shape=(2,)

        if loss_bank_rgb == 0.0 or loss_bank_ir == 0.0 or loss_bank_depth == 0.0:
            out.fill_(init_weight)
        else:
            mean = (loss_bank_rgb + loss_bank_ir + loss_bank_depth) / 3.0

            aa = rgb_w / (mean + 1e-6)
            bb = depth_w / (mean + 1e-6)
            cc = ir_w / (mean + 1e-6)

            aa = alpha - beta * torch.sigmoid((aa - gamma) * sigma)
            bb = alpha - beta * torch.sigmoid((bb - gamma) * sigma)
            cc = alpha - beta * torch.sigmoid((cc - gamma) * sigma)

            out[0] = aa
            out[1] = bb
            out[2] = cc

        return out



class SURF_MV(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]



        x_mean = (x_rgb + x_ir + x_depth) / torch.sum(p, dim=[1])

        # print(torch.sum((p)))

        x_var = torch.zeros_like(x_mean)
        if torch.sum((p)) == 1:
            x_var = torch.zeros_like(x_mean)
        else:
            for i in range(3):
                x_var += (x[i] - x_mean) ** 2
            x_var = x_var / torch.sum(p, dim=[1])
            p_sum = torch.sum(p, dim=[1, 2, 3, 4])
            # print(p_sum)
            x_var[p_sum == 1, :, :, :] = 0

        # print(torch.sum(x_mean), torch.sum(x_var))

        x_mean = x_mean.float().cuda()
        x_var = x_var.float().cuda()
        x = torch.cat((x_mean, x_var), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4


class SURF_MV_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)

        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        if args.buffer:
            self.auxi_bone = nn.Sequential(
                nn.Conv2d(args.inplace_new, args.inplace_new, 1, 1),
                model_resnet18_se_4.layer3_new,
                model_resnet18_se_4.layer4,
                model_resnet18_se_4.avgpool,
                Flatten(1),
                model_resnet18_se_4.fc,
            )
        else:
            self.auxi_bone = nn.Sequential(
                model_resnet18_se_4.layer3_new,
                model_resnet18_se_4.layer4,
                model_resnet18_se_4.avgpool,
                Flatten(1),
                model_resnet18_se_4.fc,
            )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]


        x_mean = (x_rgb + x_ir + x_depth) / torch.sum(p, dim=[1])

        # print(torch.sum((p)))

        x_var = torch.zeros_like(x_mean)
        if torch.sum((p)) == 1:
            x_var = torch.zeros_like(x_mean)
        else:
            for i in range(3):
                x_var += (x[i] - x_mean) ** 2
            x_var = x_var / torch.sum(p, dim=[1])
            p_sum = torch.sum(p, dim=[1, 2, 3, 4])
            # print(p_sum)
            x_var[p_sum == 1, :, :, :] = 0

        # print(torch.sum(x_mean), torch.sum(x_var))

        x_mean = x_mean.float().cuda()
        x_var = x_var.float().cuda()
        x = torch.cat((x_mean, x_var), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p

