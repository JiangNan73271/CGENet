import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class LightweightRDSP(nn.Module):
    """轻量级RDSP模块，用于单尺度特征增强"""
    def __init__(self, in_channel, out_channel):
        super(LightweightRDSP, self).__init__()
        self.relu = nn.ReLU(True)
        
        # 减少分支数量，使用2个分支而不是4个
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
        )
        
        self.conv_cat = BasicConv2d(2 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        
        x_cat = self.conv_cat(torch.cat((x0, x1), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class MultiScaleRDSP(nn.Module):
    """多尺度RDSP模块，直接融合三个尺度的特征"""
    def __init__(self, in_channel, out_channel, scales=3):
        super(MultiScaleRDSP, self).__init__()
        self.scales = scales
        self.relu = nn.ReLU(True)
        
        # 为每个尺度创建独立的分支
        self.scale_branches = nn.ModuleList()
        
        # 每个尺度对应不同的膨胀率
        dilation_rates = [1, 3, 5]  # 对应 s, m, l 尺度
        
        for i in range(scales):
            branch = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
                BasicConv2d(out_channel, out_channel, kernel_size=3, 
                           padding=dilation_rates[i], dilation=dilation_rates[i])
            )
            self.scale_branches.append(branch)
        
        # 跨尺度交互分支
        self.cross_scale_conv = nn.Sequential(
            BasicConv2d(in_channel * scales, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )
        
        # 最终融合
        self.conv_cat = BasicConv2d(out_channel * (scales + 1), out_channel, 3, padding=1)
        
        # 残差连接 - 对拼接后的特征
        self.conv_res = BasicConv2d(in_channel * scales, out_channel, 1)
        
    def forward(self, features):
        """
        features: list of [s_feat, m_feat, l_feat]
        所有特征应该已经调整到相同的空间尺寸
        """
        # 将特征拼接作为输入
        concat_features = torch.cat(features, dim=1)
        
        # 每个尺度的独立处理
        scale_outputs = []
        for i, feat in enumerate(features):
            scale_out = self.scale_branches[i](feat)
            scale_outputs.append(scale_out)
        
        # 跨尺度交互
        cross_scale = self.cross_scale_conv(concat_features)
        
        # 融合所有分支
        all_features = scale_outputs + [cross_scale]
        fused = self.conv_cat(torch.cat(all_features, dim=1))
        
        # 残差连接
        residual = self.conv_res(concat_features)
        
        return self.relu(fused + residual)


class HybridMultiScaleFusion(nn.Module):
    """混合多尺度融合模块 - 结合轻量级增强和多尺度融合"""
    def __init__(self, in_channels, out_channels, use_lightweight_enhance=True):
        super(HybridMultiScaleFusion, self).__init__()
        self.use_lightweight_enhance = use_lightweight_enhance
        
        if use_lightweight_enhance:
            # 先对每个尺度进行轻量级增强
            self.enhance_s = LightweightRDSP(in_channels, in_channels)
            self.enhance_m = LightweightRDSP(in_channels, in_channels)
            self.enhance_l = LightweightRDSP(in_channels, in_channels)
        
        # 然后进行多尺度融合
        self.multi_scale_rdsp = MultiScaleRDSP(in_channels, out_channels, scales=3)
        
    def forward(self, l, m, s):
        """
        l, m, s: 三个不同尺度的特征
        注意：输入的特征可能有不同的空间尺寸
        s是没有放缩的原始特征，作为基准尺寸
        """
        # 首先将所有特征调整到相同尺寸（以s为基准）
        target_size = s.shape[2:]
        
        l_resized = F.interpolate(l, size=target_size, mode='bilinear', align_corners=False) if l.shape[2:] != target_size else l
        m_resized = F.interpolate(m, size=target_size, mode='bilinear', align_corners=False) if m.shape[2:] != target_size else m
        
        # 可选的轻量级增强
        if self.use_lightweight_enhance:
            l_enhanced = self.enhance_l(l_resized)
            m_enhanced = self.enhance_m(m_resized)
            s_enhanced = self.enhance_s(s)
            features = [s_enhanced, m_enhanced, l_enhanced]
        else:
            features = [s, m_resized, l_resized]
        
        # 多尺度融合
        fused = self.multi_scale_rdsp(features)
        
        return fused


class ImprovedRDSP(nn.Module):
    """改进的RDSP模块，专门用于伪装目标检测的多尺度特征融合"""
    def __init__(self, in_channel, out_channel):
        super(ImprovedRDSP, self).__init__()
        self.relu = nn.ReLU(True)
        
        # 主干分支 - 保持原始特征
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        
        # 细节分支 - 捕获精细纹理
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=2, dilation=2)
        )
        
        # 上下文分支 - 捕获中等范围上下文
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=4, dilation=4)
        )
        
        # 全局分支 - 捕获大范围上下文
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=8, dilation=8)
        )
        
        # 自适应特征融合
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * out_channel, 4, 1),
            nn.Sigmoid()
        )
        
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        # 自适应加权融合
        cat_features = torch.cat((x0, x1, x2, x3), 1)
        attention = self.attention(cat_features)
        
        # 应用注意力权重
        weighted_features = []
        for i, feat in enumerate([x0, x1, x2, x3]):
            weight = attention[:, i:i+1, :, :]
            weighted_features.append(feat * weight)
        
        x_cat = self.conv_cat(torch.cat(weighted_features, 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


# 使用示例
class EnhancedMultiScaleFusion(nn.Module):
    """增强的多尺度融合模块，可以替换原有的SimpleConcatFusion"""
    def __init__(self, in_channels, out_channels, fusion_type='hybrid'):
        super(EnhancedMultiScaleFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'hybrid':
            self.fusion = HybridMultiScaleFusion(in_channels, out_channels, use_lightweight_enhance=True)
        elif fusion_type == 'direct':
            self.fusion = MultiScaleRDSP(in_channels, out_channels, scales=3)
        elif fusion_type == 'separate':
            # 设想1的实现
            self.rdsp_s = ImprovedRDSP(in_channels, in_channels)
            self.rdsp_m = ImprovedRDSP(in_channels, in_channels)
            self.rdsp_l = ImprovedRDSP(in_channels, in_channels)
            self.final_fusion = BasicConv2d(in_channels * 3, out_channels, 3, padding=1)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
    def forward(self, l, m, s):
        if self.fusion_type in ['hybrid', 'direct']:
            return self.fusion(l, m, s)
        else:  # separate
            # 调整到相同尺寸（以s为基准）
            target_size = s.shape[2:]
            l_resized = F.interpolate(l, size=target_size, mode='bilinear', align_corners=False) if l.shape[2:] != target_size else l
            m_resized = F.interpolate(m, size=target_size, mode='bilinear', align_corners=False) if m.shape[2:] != target_size else m
            
            # 分别增强
            s_enhanced = self.rdsp_s(s)
            m_enhanced = self.rdsp_m(m_resized)
            l_enhanced = self.rdsp_l(l_resized)
            
            # 融合
            fused = torch.cat([s_enhanced, m_enhanced, l_enhanced], dim=1)
            return self.final_fusion(fused)
