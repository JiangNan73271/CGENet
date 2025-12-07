import abc
import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


from methods.backbone.pvt_v2_eff import pvt_v2_eff_b1, pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from methods.cgenet.ops import ConvBNReLU, PixelNormalizer, resize_to
from utils.box.CT import RDSP
from utils.box.GLP import CoordAtt, conv3x3_bn_relu
from .layers import SimpleASPP

from utils.box.loss_function.utils import get_coef, cal_ual
from methods.cgenet.enhanced_layers import MultiHeadMyNet


LOGGER = logging.getLogger("main")


class _CGENet_Base(nn.Module):

    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        if self.training:
            # 训练模式：获取多层预测结果
            predictions = self.body(data=data)  # 返回 (main_out, p1, p2, p3)
            out = predictions
            mask = data["mask"]
            prob = out.sigmoid()

            # 🔧 添加NaN检测和处理，防止污染整个网络
            def safe_loss(loss_tensor, loss_name, fallback_value=0.0):
                """安全的损失处理，检测并处理NaN"""
                if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
                    print(f"警告: {loss_name} 包含NaN/Inf，使用fallback值 {fallback_value}")
                    return torch.tensor(fallback_value, device=loss_tensor.device, requires_grad=True)
                return loss_tensor


        else:
            # 推理模式：只获取主预测
            out = self.body(data=data)
            return out

        if self.training:

            def iou_loss(pred, mask):
                pred_sigmoid = torch.sigmoid(pred)
                inter = (pred_sigmoid * mask).sum(dim=(2, 3))
                union = (pred_sigmoid + mask).sum(dim=(2, 3))
                iou = 1 - (inter + 1) / (union - inter + 1)
                return iou.mean()

            def structure_loss(pred, mask):
                # 限制预测值范围，防止极端logits导致数值问题
                pred = torch.clamp(pred, min=-10.0, max=10.0)

                # 计算权重：边缘区域权重更高
                weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

                # 加权BCE损失
                wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
                # 防止除零
                weit_sum = weit.sum(dim=(2, 3)).clamp(min=1e-7)
                wbce = (weit * wbce).sum(dim=(2, 3)) / weit_sum

                # 加权IOU损失
                pred_sigmoid = torch.sigmoid(pred)
                inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
                union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
                # 防止除零和数值不稳定
                union_safe = union - inter + 1e-7
                wiou = 1 - (inter + 1e-7) / union_safe

                # 检查中间结果
                if torch.isnan(wbce).any() or torch.isnan(wiou).any():
                    print("警告: structure_loss内部计算出现NaN")
                    return torch.tensor(0.0, device=pred.device, requires_grad=True)

                return (wbce + wiou).mean()

            losses = []
            loss_str = []


            if self.use_structure_loss:
                # 主要结构化损失
                main_loss = structure_loss(out, mask)
                main_loss = safe_loss(main_loss, "main_structure_loss")
                losses.append(main_loss)
                loss_str.append(f"main_struct: {main_loss.item():.5f}")

            else:
                # 主要BCE损失
                main_bce = F.binary_cross_entropy_with_logits(input=out, target=mask, reduction="mean")
                main_bce = safe_loss(main_bce, "main_bce_loss")
                losses.append(main_bce)
                loss_str.append(f"main_bce: {main_bce.item():.5f}")

            ual_coef = get_coef(iter_percentage=iter_percentage, method='cos')
            ual_loss = cal_ual(seg_logits=out, seg_gts=mask)  # 使用logits而不是prob
            ual_loss *= ual_coef
            ual_loss = safe_loss(ual_loss, "ual_loss")  # 安全处理
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

            vis_dict = {
                "sal": prob,
            }

            return dict(vis=vis_dict, loss=sum(losses), loss_str=" ".join(loss_str))
        else:
            return out

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class RN50_CGENet(_CGENet_Base):
    def __init__(
            self, pretrained=True, num_frames=1, input_norm=True, mid_dim=64, siu_groups=4, hmu_groups=6,
            use_checkpoint=False,
            use_structure_loss=True,
            loss_weights=None,  # 🔧 损失权重配置
            **kwargs
    ):
        super().__init__()
        self.encoder = timm.create_model(
            model_name="resnet50", features_only=True, out_indices=range(5), pretrained=False
        )
        if pretrained:
            # 🔧 从本地路径加载预训练权重
            local_weight_path = "/home/ygq/wyh/CGENet/pretrained_weight/resnet50-timm.pth"
            params = torch.load(local_weight_path, map_location="cpu")
            self.encoder.load_state_dict(params, strict=False)

        # 🔧 损失函数配置
        self.use_structure_loss = use_structure_loss
        if loss_weights is None:
            # 默认权重配置（基于训练日志分析）
            self.loss_weights = {
                'bound': 1.0,
                'structure': 1.0,
                'bce': 1.0,  # BCE损失权重（当不使用结构化损失时）
                'iou': 1.0,  # IOU损失权重（当不使用结构化损失时）
            }
        else:
            self.loss_weights = loss_weights

        # self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.tra_5 = ConvBNReLU(2048, mid_dim, 3, 1, 1)
        self.tra_4 = ConvBNReLU(1024, mid_dim, 3, 1, 1)
        self.tra_3 = ConvBNReLU(512, mid_dim, 3, 1, 1)
        self.tra_2 = ConvBNReLU(256, mid_dim, 3, 1, 1)
        self.tra_1 = ConvBNReLU(64, mid_dim, 3, 1, 1)

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

        self.siu_5 = MultiHeadMyNet(mid_dim,num_heads=4)
        self.siu_4 = MultiHeadMyNet(mid_dim,num_heads=4)
        self.siu_3 = MultiHeadMyNet(mid_dim,num_heads=4)
        self.siu_2 = MultiHeadMyNet(mid_dim,num_heads=4)
        self.siu_1 = MultiHeadMyNet(mid_dim,num_heads=4)

        # #Decoder
        # self.Up1 = up_conv(ch_in=mid_dim, ch_out=mid_dim)
        # self.Up2 = up_conv(ch_in=mid_dim, ch_out=mid_dim)
        # self.Up3 = up_conv(ch_in=mid_dim, ch_out=mid_dim)
        # self.Up4 = up_conv(ch_in=mid_dim, ch_out=mid_dim)
        #
        # # 🔧 FRDSP融合感知解码器模块 - 集成fusion_conv和RDSP的优势
        # # 输入通道数为 mid_dim*2 (上采样特征 + 跳跃连接特征)
        # # 输出通道数为 mid_dim
        # self.FRDSP1 = FRDSP(mid_dim * 2, mid_dim)  # 第一层解码融合
        # self.FRDSP2 = FRDSP(mid_dim * 2, mid_dim)  # 第二层解码融合
        # self.FRDSP3 = FRDSP(mid_dim * 2, mid_dim)  # 第三层解码融合
        # self.FRDSP4 = FRDSP(mid_dim * 2, mid_dim)  # 第四层解码融合
        #
        # self.SCF = SimpleConvFusion(mid_dim * 2, mid_dim)
        #
        # # 🔧 优化的FPN特征传递门控机制
        # # 为跨尺度特征传递添加可学习权重，避免过强注入
        # self.fpn_gate_1_2 = nn.Parameter(torch.ones(1))  # f1 -> f2 的权重门控
        # self.fpn_gate_1_3 = nn.Parameter(torch.ones(1))  # f1 -> f3 的权重门控
        # self.fpn_gate_1_4 = nn.Parameter(torch.ones(1))  # f1 -> f4 的权重门控
        # self.fpn_gate_1_5 = nn.Parameter(torch.ones(1))  # f1 -> f5 的权重门控
        # self.fpn_gate_2_3 = nn.Parameter(torch.ones(1))  # f2 -> f3 的权重门控
        # self.fpn_gate_2_4 = nn.Parameter(torch.ones(1))  # f2 -> f4 的权重门控
        # self.fpn_gate_2_5 = nn.Parameter(torch.ones(1))  # f2 -> f5 的权重门控
        # self.fpn_gate_3_4 = nn.Parameter(torch.ones(1))  # f3 -> f4 的权重门控
        # self.fpn_gate_3_5 = nn.Parameter(torch.ones(1))  # f3 -> f5 的权重门控
        # self.fpn_gate_4_5 = nn.Parameter(torch.ones(1))  # f4 -> f5 的权重门控
        #
        # # FPN特征对齐层（可选，当需要更好的特征对齐时使用）
        # self.fpn_align_1_2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_1_3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_1_4 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_1_5 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_2_3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_2_4 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_2_5 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_3_4 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_3_5 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        # self.fpn_align_4_5 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        c1, c2, c3, c4, c5 = self.encoder(x)
        return c1, c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])  # [8, 3, 576, 576]
        m_trans_feats = self.normalize_encoder(data["image_m"])  # [8, 3, 480, 480]
        s_trans_feats = self.normalize_encoder(data["image_s"])  # [8, 3, 384, 384]

        l1, m1, s1 = self.tra_5(l_trans_feats[4]), self.tra_5(m_trans_feats[4]), self.tra_5(s_trans_feats[4])
        f1 = self.siu_5(l=l1, m=m1, s=s1)  # [8, 64, 12, 12] (mid_dim)
        l2, m2, s2 = self.tra_4(l_trans_feats[3]), self.tra_4(m_trans_feats[3]), self.tra_4(s_trans_feats[3])
        f2 = self.siu_4(l=l2, m=m2, s=s2)  # [8, 64, 24, 24] (mid_dim)
        l3, m3, s3 = self.tra_3(l_trans_feats[2]), self.tra_3(m_trans_feats[2]), self.tra_3(s_trans_feats[2])
        f3 = self.siu_3(l=l3, m=m3, s=s3)  # [8, 64, 48, 48] (mid_dim)
        l4, m4, s4 = self.tra_2(l_trans_feats[1]), self.tra_2(m_trans_feats[1]), self.tra_2(s_trans_feats[1])
        f4 = self.siu_2(l=l4, m=m4, s=s4)  # [8, 64, 96, 96] (mid_dim)
        l5, m5, s5 = self.tra_1(l_trans_feats[0]), self.tra_1(l_trans_feats[0]), self.tra_1(s_trans_feats[0])
        f5 = self.siu_1(l=l5, m=m5, s=s5)

        # # Decoder🔧 优化的FPN多尺度特征融合 - 加入门控机制，更适合伪装目标检测
        # # 伪装目标需要多尺度信息进行准确检测，但需要避免过强的特征注入导致信息冲突
        #
        # # 多尺度插值：将深层特征传播到浅层，并通过对齐层进行特征对齐
        # mf1_2 = F.interpolate(f1, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        # mf1_2 = self.fpn_align_1_2(mf1_2)  # 特征对齐
        #
        # mf1_3 = F.interpolate(f1, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        # mf1_3 = self.fpn_align_1_3(mf1_3)  # 特征对齐
        #
        # mf1_4 = F.interpolate(f1, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # mf1_4 = self.fpn_align_1_4(mf1_4)  # 特征对齐
        #
        # mf1_5 = F.interpolate(f1, size=f5.shape[-2:], mode='bilinear', align_corners=False)
        # mf1_5 = self.fpn_align_1_5(mf1_5)  # 特征对齐
        #
        # mf2_3 = F.interpolate(f2, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        # mf2_3 = self.fpn_align_2_3(mf2_3)  # 特征对齐
        #
        # mf2_4 = F.interpolate(f2, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # mf2_4 = self.fpn_align_2_4(mf2_4)  # 特征对齐
        #
        # mf2_5 = F.interpolate(f2, size=f5.shape[-2:], mode='bilinear', align_corners=False)
        # mf2_5 = self.fpn_align_2_5(mf2_5)  # 特征对齐
        #
        # mf3_4 = F.interpolate(f3, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # mf3_4 = self.fpn_align_3_4(mf3_4)  # 特征对齐
        #
        # mf3_5 = F.interpolate(f3, size=f5.shape[-2:], mode='bilinear', align_corners=False)
        # mf3_5 = self.fpn_align_3_5(mf3_5)  # 特征对齐
        #
        # mf4_5 = F.interpolate(f4, size=f5.shape[-2:], mode='bilinear', align_corners=False)
        # mf4_5 = self.fpn_align_4_5(mf4_5)  # 特征对齐
        #
        # enhanced_f1 = f1  # 最深层保持原样
        # enhanced_f2 = f2 + self.fpn_gate_1_2 * mf1_2 # 融合来自f1的语义信息
        # enhanced_f3 = f3 + self.fpn_gate_1_3 * mf1_3 + self.fpn_gate_2_3 * mf2_3 # 融合来自f1,f2的多尺度信息
        # enhanced_f4 = f4 + self.fpn_gate_1_4 * mf1_4 + self.fpn_gate_2_4 * mf2_4 + self.fpn_gate_3_4 * mf3_4 # 融合来自f1,f2,f3的多尺度信息
        # enhanced_f5 = f5 + self.fpn_gate_1_5 * mf1_5 + self.fpn_gate_2_5 * mf2_5 + self.fpn_gate_3_5 * mf3_5 + self.fpn_gate_4_5 * mf4_5 # 融合来自f1,f2,f3,f4的多尺度信息
        #
        # # 🔧 FRDSP融合感知解码器 - 在FPN增强特征基础上进行智能融合解码
        # # FRDSP集成了fusion_conv的分组融合和RDSP的多尺度上下文增强
        # # 第一层解码：f1(深层) -> f2尺度
        # mf1 = self.Up1(enhanced_f1)  # 上采样到f2尺度: [B, mid_dim, H2, W2]
        # mf1_concat = torch.cat((mf1, enhanced_f2), dim=1)  # 拼接: [B, 2*mid_dim, H2, W2]
        # mf1 = self.FRDSP1(mf1_concat)  # FRDSP融合: [B, mid_dim, H2, W2]
        #
        # # 第二层解码：mf1 -> f3尺度
        # mf2 = self.Up2(mf1)  # 上采样到f3尺度: [B, mid_dim, H3, W3]
        # mf2_concat = torch.cat((mf2, enhanced_f3), dim=1)  # 拼接: [B, 2*mid_dim, H3, W3]
        # mf2 = self.FRDSP2(mf2_concat)  # FRDSP融合: [B, mid_dim, H3, W3]
        #
        # # 第三层解码：mf2 -> f4尺度
        # mf3 = self.Up3(mf2)  # 上采样到f4尺度: [B, mid_dim, H4, W4]
        # mf3_concat = torch.cat((mf3, enhanced_f4), dim=1)  # 拼接: [B, 2*mid_dim, H4, W4]
        # mf3 = self.FRDSP3(mf3_concat)  # FRDSP融合: [B, mid_dim, H4, W4]
        #
        # # 第四层解码：mf3 -> f5尺度
        # mf4 = self.Up4(mf3)  # 上采样到f5尺度: [B, mid_dim, H5, W5]
        # mf4_concat = torch.cat((mf4, enhanced_f5), dim=1)  # 拼接: [B, 2*mid_dim, H5, W5]
        # mf4 = self.FRDSP4(mf4_concat)  # FRDSP融合: [B, mid_dim, H5, W5]
        #
        # out4 = self.predictor(mf4)

        target_size = f5.shape[-2:]  # 使用最浅层的空间尺寸作为目标尺寸

        mf1_5 = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        mf2_5 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        mf3_5 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        mf4_5 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)

        mf5 = mf1_5 + mf2_5 + mf3_5 + mf4_5 + f5
        mf5 = self.predictor(mf5)
        return mf5

        # return out4


class FRDSP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FRDSP, self).__init__()
        assert in_channels == 2 * out_channels, f"期望输入通道数为输出的2倍，得到 {in_channels} vs {out_channels}"

        self.out_channels = out_channels

        # 1. 源内特征提取（基于fusion_conv的分组卷积思想）
        # 对解码器上采样特征和编码器跳跃特征分别进行逐源提取
        groups = 2  # 固定2组：[decoder_features, encoder_skip_features]
        self.source_extraction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(in_channels)
        )

        # 2. 倒置瓶颈跨源融合（基于fusion_conv的逐点卷积思想）
        expansion_factor = 4
        mid_channels = out_channels * expansion_factor

        self.cross_fusion = nn.Sequential(
            # Expand - 扩张通道以增强表达能力
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(mid_channels),
            # Project - 压缩回目标通道数
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

        # 3. 原始RDSP多尺度上下文增强（遵循不改变原则）
        self.rdsp_context = RDSP(out_channels, out_channels)

        # 4. 整体残差连接的1x1对齐层
        self.residual_align = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 5. 可选的轻量注意力（通道注意力）
        self.channel_attention = ChannelAttention(out_channels)

        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入: [batch, 2*out_channels, H, W] = [解码器特征 + 编码器跳跃特征]
        identity = x

        # 1. 源内特征提取 - 分别处理两个来源的特征，避免过早混合
        x = self.source_extraction(x)  # [B, 2*C, H, W]

        # 2. 跨源融合 - 高效密集的特征融合
        x = self.cross_fusion(x)  # [B, C, H, W]

        # 保存融合结果用于整体残差
        fusion_result = x

        # 3. RDSP多尺度上下文增强
        x = self.rdsp_context(x)  # [B, C, H, W]

        # 4. 通道注意力
        x = self.channel_attention(x)

        # 5. 整体残差连接：原始输入经1x1对齐后与RDSP输出相加
        residual = self.residual_align(identity)  # [B, 2*C, H, W] -> [B, C, H, W]
        x = x + residual

        # 6. 最终激活
        x = self.final_activation(x)

        return x


class ChannelAttention(nn.Module):
    """
    轻量级通道注意力模块
    """

    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# 🔬 消融实验模块 - 方案三的不同FRDSP替换方案

class SimpleConvFusion(nn.Module):
    """
    方案三-1: 普通卷积替换FRDSP
    带有跳跃连接的3×3卷积层，类似于ResBlock
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleConvFusion, self).__init__()
        assert in_channels == 2 * out_channels, f"期望输入通道数为输出的2倍，得到 {in_channels} vs {out_channels}"
        
        # 简单的3x3卷积融合 + 残差连接
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 残差连接的1x1对齐层
        self.residual_align = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.final_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 简单的卷积融合
        conv_out = self.conv_fusion(x)
        
        # 残差连接
        residual = self.residual_align(x)
        
        # 输出
        out = self.final_activation(conv_out + residual)
        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class PvtV2B2_CGENet(_CGENet_Base):
    def __init__(
            self,
            pretrained=True,
            num_frames=1,
            input_norm=True,
            mid_dim=64,
            siu_groups=4,
            hmu_groups=6,
            use_checkpoint=False,
            use_structure_loss=True,
            loss_weights=None,  # 🔧 损失权重配置
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        # 🔧 损失函数配置
        self.use_structure_loss = use_structure_loss
        if loss_weights is None:
            # 默认权重配置（基于训练日志分析）
            self.loss_weights = {
                'bound': 0.25,
                'structure': 1.0,
                'bce': 1.0,  # BCE损失权重（当不使用结构化损失时）
                'iou': 1.0,  # IOU损失权重（当不使用结构化损失时）
            }
        else:
            self.loss_weights = loss_weights

        self.embed_dims = self.encoder.embed_dims
        # self.tra_5 = EFF_ImprovedSA(self.embed_dims[3], out_dim=mid_dim, is_bottom=True)
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        # self.tra_5 = ConvBNReLU(self.embed_dims[3], mid_dim, 3, 1, 1)
        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

        # self.predictor_64 = nn.Sequential(
        #     # ConvBNReLU(128, 64, 3, 1, 1),
        #     ConvBNReLU(64, 32, 3, 1, 1),
        #     nn.Conv2d(32, 1, 1),
        # )

        self.dwc1 = conv3x3_bn_relu(512, 320, k=1, s=1, p=0)
        self.dwc2 = conv3x3_bn_relu(320, 128, k=1, s=1, p=0)
        self.dwc3 = conv3x3_bn_relu(128, 64, k=1, s=1, p=0)
        self.dwcon_2 = conv3x3_bn_relu(320, 320)
        self.dwcon_3 = conv3x3_bn_relu(128, 128)
        self.dwcon_4 = conv3x3_bn_relu(64, 64)

        # self.siu_5 = SimpleConcatFusion(in_channels=mid_dim, out_channels=mid_dim)
        # self.siu_4 = SimpleConcatFusion(in_channels=mid_dim, out_channels=mid_dim)
        # self.siu_3 = SimpleConcatFusion(in_channels=mid_dim, out_channels=mid_dim)
        # self.siu_2 = SimpleConcatFusion(in_channels=mid_dim, out_channels=mid_dim)

        self.siu_5 = MultiHeadMyNet(mid_dim, num_heads=4)
        self.siu_4 = MultiHeadMyNet(mid_dim, num_heads=4)
        self.siu_3 = MultiHeadMyNet(mid_dim, num_heads=4)
        self.siu_2 = MultiHeadMyNet(mid_dim, num_heads=4)

        # Decoder
        self.Up1 = up_conv(ch_in=mid_dim, ch_out=mid_dim)
        self.Up2 = up_conv(ch_in=mid_dim, ch_out=mid_dim)
        self.Up3 = up_conv(ch_in=mid_dim, ch_out=mid_dim)


        # 输入通道数为 mid_dim*2 (上采样特征 + 跳跃连接特征)
        # 输出通道数为 mid_dim
        self.FRDSP1 = FRDSP(mid_dim * 2, mid_dim)  # 第一层解码融合
        self.FRDSP2 = FRDSP(mid_dim * 2, mid_dim)  # 第二层解码融合
        self.FRDSP3 = FRDSP(mid_dim * 2, mid_dim)  # 第三层解码融合

        # 🔧 优化的FPN特征传递门控机制
        # 为跨尺度特征传递添加可学习权重，避免过强注入
        self.fpn_gate_1_2 = nn.Parameter(torch.ones(1))  # f1 -> f2 的权重门控
        self.fpn_gate_1_3 = nn.Parameter(torch.ones(1))  # f1 -> f3 的权重门控
        self.fpn_gate_1_4 = nn.Parameter(torch.ones(1))  # f1 -> f4 的权重门控
        self.fpn_gate_2_3 = nn.Parameter(torch.ones(1))  # f2 -> f3 的权重门控
        self.fpn_gate_2_4 = nn.Parameter(torch.ones(1))  # f2 -> f4 的权重门控
        self.fpn_gate_3_4 = nn.Parameter(torch.ones(1))  # f3 -> f4 的权重门控

        # FPN特征对齐层（可选，当需要更好的特征对齐时使用）
        self.fpn_align_1_2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        self.fpn_align_1_3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        self.fpn_align_1_4 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        self.fpn_align_2_3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        self.fpn_align_2_4 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        self.fpn_align_3_4 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]  # l:[8, 64, 144, 144] m:[8, 64, 96, 96]  s:[8, 64, 120, 120]
        c3 = features["reduction_3"]  # l:[8, 128, 72, 72]  m:[8, 128, 48, 48] s:[8, 128, 60, 60]
        c4 = features["reduction_4"]  # l:[8, 320, 36, 36]  m:[8, 320, 24, 24] s:[8, 320, 30, 30]
        c5 = features["reduction_5"]  # l:[8, 512, 18, 18]  m:[8, 512, 12, 12] s:[8, 512, 15, 15]
        return c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])  # [8, 3, 576, 576]
        m_trans_feats = self.normalize_encoder(data["image_m"])  # [8, 3, 480, 480]
        s_trans_feats = self.normalize_encoder(data["image_s"])  # [8, 3, 384, 384]

        l1, m1, s1 = self.tra_5(l_trans_feats[3]), self.tra_5(m_trans_feats[3]), self.tra_5(s_trans_feats[3])
        f1 = self.siu_5(l=l1, m=m1, s=s1)  # [8, 64, 12, 12] (mid_dim)
        l2, m2, s2 = self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2]), self.tra_4(s_trans_feats[2])
        f2 = self.siu_4(l=l2, m=m2, s=s2)  # [8, 64, 24, 24] (mid_dim)
        l3, m3, s3 = self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1]), self.tra_3(s_trans_feats[1])
        f3 = self.siu_3(l=l3, m=m3, s=s3)  # [8, 64, 48, 48] (mid_dim)
        l4, m4, s4 = self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0]), self.tra_2(s_trans_feats[0])
        f4 = self.siu_2(l=l4, m=m4, s=s4)  # [8, 64, 96, 96] (mid_dim)
        # l1, s1 = self.tra_5(l_trans_feats[3]), self.tra_5(s_trans_feats[3])
        # f1 = self.siu_5(l=l1, s=s1)  # [8, 64, 12, 12] (mid_dim)
        # l2, s2 = self.tra_4(l_trans_feats[2]), self.tra_4(s_trans_feats[2])
        # f2 = self.siu_4(l=l2, s=s2)  # [8, 64, 24, 24] (mid_dim)
        # l3, s3 = self.tra_3(l_trans_feats[1]), self.tra_3(s_trans_feats[1])
        # f3 = self.siu_3(l=l3,  s=s3)  # [8, 64, 48, 48] (mid_dim)
        # l4, s4 = self.tra_2(l_trans_feats[0]), self.tra_2(s_trans_feats[0])
        # f4 = self.siu_2(l=l4, s=s4)  # [8, 64, 96, 96] (mid_dim)

        # 🔧 优化的FPN多尺度特征融合 - 加入门控机制，更适合伪装目标检测
        # 伪装目标需要多尺度信息进行准确检测，但需要避免过强的特征注入导致信息冲突

        # 多尺度插值：将深层特征传播到浅层，并通过对齐层进行特征对齐
        mf1_2 = F.interpolate(f1, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        mf1_2 = self.fpn_align_1_2(mf1_2)  # 特征对齐

        mf1_3 = F.interpolate(f1, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        mf1_3 = self.fpn_align_1_3(mf1_3)  # 特征对齐

        mf1_4 = F.interpolate(f1, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        mf1_4 = self.fpn_align_1_4(mf1_4)  # 特征对齐

        mf2_3 = F.interpolate(f2, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        mf2_3 = self.fpn_align_2_3(mf2_3)  # 特征对齐

        mf2_4 = F.interpolate(f2, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        mf2_4 = self.fpn_align_2_4(mf2_4)  # 特征对齐

        mf3_4 = F.interpolate(f3, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        mf3_4 = self.fpn_align_3_4(mf3_4)  # 特征对齐

        enhanced_f1 = f1  # 最深层保持原样
        enhanced_f2 = f2 + self.fpn_gate_1_2 * mf1_2  # 门控融合来自f1的语义信息
        enhanced_f3 = f3 + self.fpn_gate_1_3 * mf1_3 + self.fpn_gate_2_3 * mf2_3  # 门控融合多尺度信息
        enhanced_f4 = f4 + self.fpn_gate_1_4 * mf1_4 + self.fpn_gate_2_4 * mf2_4 + self.fpn_gate_3_4 * mf3_4  # 门控融合所有尺度信息


        # 第一层解码：f1(深层) -> f2尺度
        mf1 = self.Up1(enhanced_f1)  # 上采样到f2尺度: [B, mid_dim, H2, W2]
        mf1_concat = torch.cat((mf1, enhanced_f2), dim=1)  # 拼接: [B, 2*mid_dim, H2, W2]
        mf1 = self.FRDSP1(mf1_concat)  # FRDSP融合: [B, mid_dim, H2, W2]

        # 第二层解码：mf1 -> f3尺度
        mf2 = self.Up2(mf1)  # 上采样到f3尺度: [B, mid_dim, H3, W3]
        mf2_concat = torch.cat((mf2, enhanced_f3), dim=1)  # 拼接: [B, 2*mid_dim, H3, W3]
        mf2 = self.FRDSP2(mf2_concat)  # FRDSP融合: [B, mid_dim, H3, W3]

        # 第三层解码：mf2 -> f4尺度
        mf3 = self.Up3(mf2)  # 上采样到f4尺度: [B, mid_dim, H4, W4]
        mf3_concat = torch.cat((mf3, enhanced_f4), dim=1)  # 拼接: [B, 2*mid_dim, H4, W4]
        mf3 = self.FRDSP3(mf3_concat)  # FRDSP融合: [B, mid_dim, H4, W4]

        out4 = self.predictor(mf3)

        out4 = F.interpolate(out4, size=data["image_s"].size()[2:], mode='bilinear', align_corners=True)

        return out4
        # out = self.predictor(enhanced_f4)
        # out = F.interpolate(out, size=data["image_s"].size()[2:], mode='bilinear', align_corners=True)
        # return out

class PvtV2B1_CGENet(PvtV2B2_CGENet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b1(pretrained=pretrained, use_checkpoint=use_checkpoint)

class PvtV2B3_CGENet(PvtV2B2_CGENet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b3(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B4_CGENet(PvtV2B2_CGENet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)


class PvtV2B5_CGENet(PvtV2B2_CGENet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


class videoPvtV2B5_CGENet(PvtV2B5_CGENet):
    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "temperal_proj" in name:
                    param_groups["retrained"].append(param)
                else:
                    param_groups["pretrained"].append(param)

        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


# class EffB1_CGENet(_CGENet_Base):
#     def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64, siu_groups=4, hmu_groups=6, **kwargs):
#         super().__init__()
#         self.set_backbone(pretrained)
#
#         self.tra_5 = SimpleASPP(self.embed_dims[4], out_dim=mid_dim)
#         self.siu_5 = MHSIU(mid_dim, siu_groups)
#         self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
#
#         self.tra_4 = ConvBNReLU(self.embed_dims[3], mid_dim, 3, 1, 1)
#         self.siu_4 = MHSIU(mid_dim, siu_groups)
#         self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
#
#         self.tra_3 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
#         self.siu_3 = MHSIU(mid_dim, siu_groups)
#         self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
#
#         self.tra_2 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
#         self.siu_2 = MHSIU(mid_dim, siu_groups)
#         self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
#
#         self.tra_1 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
#         self.siu_1 = MHSIU(mid_dim, siu_groups)
#         self.hmu_1 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)
#
#         self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
#         self.predictor = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             ConvBNReLU(64, 32, 3, 1, 1),
#             nn.Conv2d(32, 1, 1),
#         )
#
#     def set_backbone(self, pretrained):
#         self.encoder = EfficientNet.from_pretrained("efficientnet-b1", pretrained=pretrained)
#         self.embed_dims = [16, 24, 40, 112, 320]
#
#     def normalize_encoder(self, x):
#         x = self.normalizer(x)
#         features = self.encoder.extract_endpoints(x)
#         c1 = features["reduction_1"]
#         c2 = features["reduction_2"]
#         c3 = features["reduction_3"]
#         c4 = features["reduction_4"]
#         c5 = features["reduction_5"]
#         return c1, c2, c3, c4, c5
#
#     def body(self, data):
#         l_trans_feats = self.normalize_encoder(data["image_l"])
#         m_trans_feats = self.normalize_encoder(data["image_m"])
#         s_trans_feats = self.normalize_encoder(data["image_s"])
#
#         l, m, s = self.tra_5(l_trans_feats[4]), self.tra_5(m_trans_feats[4]), self.tra_5(s_trans_feats[4])
#         lms = self.siu_5(l=l, m=m, s=s)
#         x = self.hmu_5(lms)
#
#         l, m, s = self.tra_4(l_trans_feats[3]), self.tra_4(m_trans_feats[3]), self.tra_4(s_trans_feats[3])
#         lms = self.siu_4(l=l, m=m, s=s)
#         x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#
#         l, m, s = self.tra_3(l_trans_feats[2]), self.tra_3(m_trans_feats[2]), self.tra_3(s_trans_feats[2])
#         lms = self.siu_3(l=l, m=m, s=s)
#         x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#
#         l, m, s = self.tra_2(l_trans_feats[1]), self.tra_2(m_trans_feats[1]), self.tra_2(s_trans_feats[1])
#         lms = self.siu_2(l=l, m=m, s=s)
#         x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#
#         l, m, s = self.tra_1(l_trans_feats[0]), self.tra_1(m_trans_feats[0]), self.tra_1(s_trans_feats[0])
#         lms = self.siu_1(l=l, m=m, s=s)
#         x = self.hmu_1(lms + resize_to(x, tgt_hw=lms.shape[-2:]))
#
#         return self.predictor(x)


# class EffB4_CGENet(EffB1_CGENet):
#     def set_backbone(self, pretrained):
#         self.encoder = EfficientNet.from_pretrained("efficientnet-b4", pretrained=pretrained)
#         self.embed_dims = [24, 32, 56, 160, 448]

if __name__ == "__main__":
    model1 = PvtV2B2_CGENet(
        pretrained=True,
        use_checkpoint=False,
        use_structure_loss=True
    ).cuda()

    # 示例2：使用分离的BCE+IOU损失
    model2 = PvtV2B2_CGENet(
        pretrained=True,
        use_checkpoint=False,
        use_structure_loss=False  # 使用分离的BCE+IOU损失
    ).cuda()


    custom_weights = {
        'bound': 0.3,
        'structure': 1.5,  # 降低结构化损失权重
        'bce': 1.0,  # BCE损失权重
        'iou': 1.2,  # 增加IOU损失权重
    }
    model3 = PvtV2B2_CGENet(
        pretrained=True,
        use_checkpoint=False,
        use_structure_loss=False,
        loss_weights=custom_weights
    ).cuda()

    # 测试前向传播
    input_tensor = torch.randn(2, 3, 384, 384).cuda()
    output = model1(input_tensor)
    print(f"Output shape: {output.shape}")

    # 模拟训练时的损失计算
    dummy_data = {
        "image_l": torch.randn(2, 3, 576, 576).cuda(),
        "image_m": torch.randn(2, 3, 384, 384).cuda(),
        "image_s": torch.randn(2, 3, 480, 480).cuda(),
        "mask": torch.randint(0, 2, (2, 1, 384, 384)).float().cuda()
    }

    model1.train()
    result = model1(dummy_data, iter_percentage=0.5)
    print(f"Loss: {result['loss'].item():.5f}")
    print(f"Loss breakdown: {result['loss_str']}")
