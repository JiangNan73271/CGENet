"""
增强的MHSIU模块实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from methods.cgenet.layers import DifferenceAwareOps
from .ops import ConvBNReLU


# ===============================================================================
# 基础注意力模块
# ===============================================================================
class MyNet(nn.Module):
    def __init__(self, in_c, num_groups=3, hidden_dim=None, num_frames=1):
        super().__init__()

        # 预处理层 - 保持各尺度的独立性
        self.conv_l_pre = ConvBNReLU(in_c, in_c, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_c, in_c, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_c, in_c, 3, 1, 1)  # intra-branch
        self.conv_m = ConvBNReLU(in_c, in_c, 3, 1, 1)  # intra-branch
        self.conv_s = ConvBNReLU(in_c, in_c, 3, 1, 1)  # intra-branch

        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c

        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(
            DifferenceAwareOps(num_frames=num_frames),
            ConvBNReLU(num_groups * hidden_dim, in_c, 3, 1, 1, act_name=None),
        )
        self.final_relu = nn.ReLU(True)

    def forward(self, l, m, s):  # expand_conv(x)[2, 192,12,12]
        tgt_size = s.shape[2:]

        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        m = self.conv_s_pre(m)
        m = F.adaptive_max_pool2d(m, tgt_size) + F.adaptive_avg_pool2d(m, tgt_size)

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)

        outs = []
        gates = []
        # 创建两个空列表，用于收集各组的输出特征和门控特征。

        group_id = 0
        branch_out = self.interact[str(group_id)](l)  # 对1.5倍放大图像-l扩展到高维度 [b, 64, h, w] -> [b, 64*3, h, w]
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)  # 划分成3个分支
        # curr_out:当前组的输出特征——g3， curr_fork:传递特征——g1， curr_gate:门控特征——g2
        outs.append(curr_out)
        gates.append(curr_gate)

        group_id = 1
        curr_m = torch.cat([m, curr_fork], dim=1)  # 创建一个新张量，将m原始特征和l传递特征进行拼接
        branch_out = self.interact[str(group_id)](curr_m)  # 对拼接图像扩展到高维度 [b, 64*2, h, w] -> [b, 64*3, h, w]
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)  # 划分成3个分支
        outs.append(curr_out)
        gates.append(curr_gate)

        group_id = 2
        curr_x = torch.cat([s, curr_fork], dim=1)  # 创建一个新张量，将s原始特征和m传递特征进行拼接
        branch_out = self.interact[str(group_id)](curr_x)  # 对拼接图像卷积 [b, 64*2, h, w] -> [b, 64*2, h, w]
        curr_out, curr_gate = branch_out.chunk(2, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        out = torch.cat(outs, dim=1)
        gate = self.gate_genator(torch.cat(gates, dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + s)


class MultiHeadMyNet(nn.Module):
    def __init__(self, in_c, num_groups=3, num_heads=4, hidden_dim=None, num_frames=1,
                 gate_type='softmax', fusion_type='conv', norm_type='bn'):
        """
        MultiHeadMyNet 模块的消融实验版本
        
        Args:
            gate_type (str): 门控方式 'softmax'|'sigmoid'|'none'
            fusion_type (str): 头间融合方式 'conv'|'weighted_sum'
            norm_type (str): 归一化类型 'bn'|'gn'
        """
        super().__init__()

        self.gate_type = gate_type
        self.fusion_type = fusion_type
        self.norm_type = norm_type

        # 根据归一化类型选择ConvNormAct函数
        if norm_type == 'bn':
            ConvNormAct = lambda in_c, out_c, k, s=1, p=None, act_name='relu': ConvBNReLU(in_c, out_c, k, s, p if p is not None else k//2, act_name=act_name)
        elif norm_type == 'gn':
            ConvNormAct = lambda in_c, out_c, k, s=1, p=None, act_name='silu': ConvGNSiLU(in_c, out_c, k, s, p if p is not None else k//2, act_name=act_name)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")

        # 保持原始的预处理层
        self.conv_l_pre = ConvNormAct(in_c, in_c, 3, 1, 1)
        self.conv_m_pre = ConvNormAct(in_c, in_c, 3, 1, 1)
        self.conv_s_pre = ConvNormAct(in_c, in_c, 3, 1, 1)

        self.conv_l = ConvNormAct(in_c, in_c, 3, 1, 1)  # intra-branch
        self.conv_m = ConvNormAct(in_c, in_c, 3, 1, 1)  # intra-branch
        self.conv_s = ConvNormAct(in_c, in_c, 3, 1, 1)  # intra-branch

        self.num_groups = num_groups
        self.num_heads = num_heads
        hidden_dim = hidden_dim or in_c

        # 当 num_heads == 0 时，走单头 MyNet 直通路径；>0 时启用多头
        self.use_multihead = (self.num_heads is not None) and (self.num_heads > 0)
        if not self.use_multihead:
            self.single_head = MyNet(in_c=in_c, num_groups=num_groups, hidden_dim=hidden_dim, num_frames=num_frames)
        else:
            # 🔧 消融1: 门控方式消融
            self.gate_generators = nn.ModuleList()
            for _ in range(num_heads):
                if gate_type == 'none':
                    # 无门控：直接返回全1权重
                    self.gate_generators.append(nn.Identity())
                else:
                    # 有门控：根据类型选择激活函数
                    gate_activation = nn.Softmax(dim=1) if gate_type == 'softmax' else nn.Sigmoid()
                    self.gate_generators.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
                        nn.ReLU(True),
                        nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
                        gate_activation,
                    ))

            # 多头交互模块 - 使用选定的归一化方式
            self.interact_heads = nn.ModuleList()
            for head in range(num_heads):
                head_interact = nn.ModuleDict()
                head_interact["0"] = ConvNormAct(hidden_dim, 3 * hidden_dim, 3, 1, 1)
                for group_id in range(1, num_groups - 1):
                    head_interact[str(group_id)] = ConvNormAct(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
                head_interact[str(num_groups - 1)] = ConvNormAct(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
                self.interact_heads.append(head_interact)

            # 🔧 消融2: 头间融合方式消融
            if fusion_type == 'conv':
                # Conv1x1融合
                self.head_fusion = ConvNormAct(num_heads * num_groups * hidden_dim, num_groups * hidden_dim, 1)
            elif fusion_type == 'weighted_sum':
                # 可学习标量加权求和
                self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)  # 初始化为均等权重
            else:
                raise ValueError(f"不支持的融合类型: {fusion_type}")

            # 保持原始的融合层 - 使用选定的归一化方式
            self.fuse = nn.Sequential(
                DifferenceAwareOps(num_frames=num_frames),
                ConvNormAct(num_groups * hidden_dim, in_c, 3, 1, 1, act_name=None),
            )
            
            # 🔧 消融3: 最终激活函数根据归一化类型选择
            if norm_type == 'bn':
                self.final_activation = nn.ReLU(True)
            elif norm_type == 'gn':
                self.final_activation = nn.SiLU(True)


    def forward(self, l, m, s):
        if not getattr(self, 'use_multihead', True):
            return self.single_head(l, m, s)

        # 保持以s尺度为基准
        tgt_size = s.shape[2:]

        # 预处理 - 与原始MyNet完全一致
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)

        m = self.conv_m_pre(m)
        m = F.adaptive_max_pool2d(m, tgt_size) + F.adaptive_avg_pool2d(m, tgt_size)

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)

        # 多头并行处理
        all_head_outs = []
        # all_head_gates = []

        for head_idx in range(self.num_heads):
            # 每个头独立执行原始MyNet的逻辑
            outs = []
            gates = []

            # Group 0: 处理l尺度
            group_id = 0
            branch_out = self.interact_heads[head_idx][str(group_id)](l)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

            # Group 1: 处理m尺度 + l的fork
            group_id = 1
            curr_m = torch.cat([m, curr_fork], dim=1)
            branch_out = self.interact_heads[head_idx][str(group_id)](curr_m)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

            # Group 2: 处理s尺度 + m的fork
            group_id = 2
            curr_x = torch.cat([s, curr_fork], dim=1)
            branch_out = self.interact_heads[head_idx][str(group_id)](curr_x)
            curr_out, curr_gate = branch_out.chunk(2, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

            # 当前头的特征拼接
            head_out = torch.cat(outs, dim=1)  # [B, num_groups * hidden_dim, H, W]
            
            # 🔧 消融1: 门控方式处理
            if self.gate_type == 'none':
                # 无门控：直接使用原始特征
                gated_head_out = head_out
            else:
                # 有门控：应用门控权重
                head_gate = self.gate_generators[head_idx](torch.cat(gates, dim=1))
                gated_head_out = head_out * head_gate

            all_head_outs.append(gated_head_out)

        # 🔧 消融2: 头间融合方式处理
        if self.fusion_type == 'conv':
            # Conv1x1融合
            multi_head_out = torch.cat(all_head_outs, dim=1)  # [B, num_heads * num_groups * hidden_dim, H, W]
            fused_out = self.head_fusion(multi_head_out)  # [B, num_groups * hidden_dim, H, W]
        elif self.fusion_type == 'weighted_sum':
            # 可学习标量加权求和
            weights = F.softmax(self.head_weights, dim=0)  # 确保权重和为1
            fused_out = sum(w * head_out for w, head_out in zip(weights, all_head_outs))

        # 最终融合和残差连接
        final_out = self.fuse(fused_out)  # [B, in_c, H, W]
        return self.final_activation(final_out + s)


# 辅助函数：GroupNorm + SiLU 版本的ConvNormAct
class ConvGNSiLU(nn.Module):
    """GroupNorm + SiLU 版本的卷积块"""

    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1, act_name='silu'):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)

        # GroupNorm: 使用16个组（或通道数的1/8，最少1组）
        num_groups = min(16, max(1, out_c // 8))
        self.norm = nn.GroupNorm(num_groups, out_c)

        if act_name == 'silu':
            self.act = nn.SiLU(True)
        elif act_name == 'relu':
            self.act = nn.ReLU(True)
        elif act_name is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"不支持的激活函数: {act_name}")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class scSE(nn.Module):

    def __init__(self, in_channel, reduction=16):
        super().__init__()
        
        # Channel Squeeze & Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, 1),
            nn.Sigmoid()
        )
        
        # Spatial Squeeze & Excitation
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel SE
        cse = self.cSE(x) * x
        
        # Spatial SE  
        sse = self.sSE(x) * x
        
        # Concurrent - take element-wise maximum
        return torch.max(cse, sse)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb):
        x = rgb

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) # 假设是[n, mip, h+w, 1]

        x_h, x_w = torch.split(y, [h, w], dim=2) # x_h: [n, mip, h, 1], [n, mip, w, 1]
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)
        out = x.mul(out_sa)
        out = self.conv_end(out)

        return out


class h_sigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish激活函数"""
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



class DSAMBlock(nn.Module):
    """
    Dual-domain Strip Attention Module (DSAM)
    简化版本，用于MHSIU增强
    """
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        
        # 立方注意力组件
        self.cubic_attention = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, 1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, in_channel, 1),
            nn.Sigmoid()
        )
        
        # 条带注意力组件
        self.strip_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 水平条带
            nn.Conv2d(in_channel, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 立方注意力
        cubic_att = self.cubic_attention(x)
        
        # 条带注意力  
        strip_att = self.strip_attention(x)
        strip_att = strip_att.expand_as(x)
        
        # 融合两种注意力
        enhanced = x * cubic_att * strip_att
        
        return enhanced


# ===============================================================================
# 增强的MHSIU模块
# ===============================================================================

class EnhancedMHSIU(nn.Module):
    """
    增强的多层次尺度集成单元（Enhanced Multi-Hierarchical Scale Integration Unit）
    
    支持的注意力类型：
    - 'scSE': 并发空间通道挤压激励注意力
    - 'coord': 坐标注意力  
    - 'hybrid': 混合注意力（scSE + CoordAtt）
    - 'original': 原始卷积注意力
    """
    
    def __init__(self, in_dim, num_groups=4, attention_type='scSE'):
        super().__init__()
        
        # 保持原有的基础结构
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        self.conv_lms = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)
        self.initial_merge = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)

        self.num_groups = num_groups
        self.attention_type = attention_type

        # 🔧 核心改进：替换原有的卷积注意力为更先进的注意力机制
        if attention_type == 'scSE':
            # 方案1：使用scSE注意力 - 推荐方案
            self.attention_module = scSE(3 * in_dim // num_groups, reduction=4)
            self.trans = nn.Sequential(
                self.attention_module,  # 🔧 先应用scSE注意力
                ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                nn.Conv2d(in_dim // num_groups, 3, 1),
                nn.Softmax(dim=1),
            )
        elif attention_type == 'coord':
            # 方案2：使用CoordAtt注意力
            self.attention_module = CoordAtt(3 * in_dim // num_groups, 3 * in_dim // num_groups)
            self.trans = nn.Sequential(
                self.attention_module,  # 🔧 先应用CoordAtt注意力
                ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                nn.Conv2d(in_dim // num_groups, 3, 1),
                nn.Softmax(dim=1),
            )
        elif attention_type == 'hybrid':
            # 方案3：混合注意力（scSE + CoordAtt）
            self.scse_module = scSE(3 * in_dim // num_groups, reduction=4)
            self.coordatt_module = CoordAtt(3 * in_dim // num_groups, 3 * in_dim // num_groups)
            # 学习两种注意力的融合权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            self.trans = nn.Sequential(
                ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                nn.Conv2d(in_dim // num_groups, 3, 1),
                nn.Softmax(dim=1),
            )
        elif attention_type == 'msca':
            # 方案4：使用MSCA多尺度卷积注意力
            try:
                from .advanced_enhanced_layers import MSCA_AttentionModule
                self.attention_module = MSCA_AttentionModule(3 * in_dim // num_groups)
                self.trans = nn.Sequential(
                    self.attention_module,  # 🔧 先应用MSCA注意力
                    ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                    nn.Conv2d(in_dim // num_groups, 3, 1),
                    nn.Softmax(dim=1),
                )
            except ImportError:
                print("警告: MSCA模块不可用，回退到原始注意力")
                self.trans = nn.Sequential(
                    ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                    nn.Conv2d(in_dim // num_groups, 3, 1),
                    nn.Softmax(dim=1),
                )
        elif attention_type == 'scsa':
            # 方案5：使用SCSA空间通道协同注意力
            try:
                from .advanced_enhanced_layers import SCSA_Simplified
                self.attention_module = SCSA_Simplified(3 * in_dim // num_groups)
                self.trans = nn.Sequential(
                    self.attention_module,  # 🔧 先应用SCSA注意力
                    ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                    nn.Conv2d(in_dim // num_groups, 3, 1),
                    nn.Softmax(dim=1),
                )
            except ImportError:
                print("警告: SCSA模块不可用，回退到原始注意力")
                self.trans = nn.Sequential(
                    ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                    nn.Conv2d(in_dim // num_groups, 3, 1),
                    nn.Softmax(dim=1),
                )
        elif attention_type == 'original':
            # 保持原始的卷积注意力
            self.trans = nn.Sequential(
                ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
                nn.Conv2d(in_dim // num_groups, 3, 1),
                nn.Softmax(dim=1),
            )
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}。支持的类型: 'scSE', 'coord', 'hybrid', 'msca', 'scsa', 'original'")

    def forward(self, l, m, s):
        # 原有的尺度对齐逻辑保持不变
        tgt_size = s.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        m = self.conv_s_pre(m)
        m = F.adaptive_max_pool2d(m, tgt_size) + F.adaptive_avg_pool2d(m, tgt_size)

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        lms = torch.cat([l, m, s], dim=1)

        # 🔧 增强的注意力处理
        attn = self.conv_lms(lms)
        attn = rearrange(attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=3, ng=self.num_groups)
        
        if self.attention_type == 'hybrid':
            # 混合注意力：应用两种注意力并融合
            scse_out = self.scse_module(attn)
            coord_out = self.coordatt_module(attn)
            
            # 学习权重融合
            weight = torch.sigmoid(self.fusion_weight)
            attn = weight * scse_out + (1 - weight) * coord_out
            
            # 应用最终变换
            attn = self.trans(attn)
        else:
            # 单一注意力或原始注意力
            attn = self.trans(attn)
        
        attn = attn.unsqueeze(dim=2)  # BTG,3,1,H,W

        # 特征融合逻辑保持不变
        x = self.initial_merge(lms)
        x = rearrange(x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=3, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)

        return x


class DSAM_EnhancedMHSIU(nn.Module):
    """
    使用DSAM注意力增强的MHSIU
    """
    
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        
        # 保持原有的基础结构
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        self.conv_lms = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)
        self.initial_merge = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)

        self.num_groups = num_groups
        
        # 使用DSAM注意力模块
        self.dsam_module = DSAMBlock(3 * in_dim // num_groups)
        
        self.trans = nn.Sequential(
            self.dsam_module,  # 🔧 先应用DSAM注意力
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, l, m, s):
        # 原有的尺度对齐逻辑
        tgt_size = s.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        m = self.conv_s_pre(m)
        m = F.adaptive_max_pool2d(m, tgt_size) + F.adaptive_avg_pool2d(m, tgt_size)

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        lms = torch.cat([l, m, s], dim=1)

        # DSAM增强的注意力处理
        attn = self.conv_lms(lms)
        attn = rearrange(attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=3, ng=self.num_groups)
        
        attn = self.trans(attn)
        attn = attn.unsqueeze(dim=2)

        # 特征融合
        x = self.initial_merge(lms)
        x = rearrange(x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=3, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)

        return x

# ===============================================================================
# 测试代码
# ===============================================================================

if __name__ == "__main__":
    print("=== 测试增强MHSIU模块 ===")
    
    # 创建测试数据
    l = torch.randn(2, 64, 36, 36)  # 大尺度
    m = torch.randn(2, 64, 24, 24)  # 中尺度  
    s = torch.randn(2, 64, 12, 12)  # 小尺度
    
    print("输入形状:")
    print(f"  大尺度 (l): {l.shape}")
    print(f"  中尺度 (m): {m.shape}")
    print(f"  小尺度 (s): {s.shape}")
    
    # 测试不同的注意力类型
    attention_types = ['scSE', 'coord', 'hybrid', 'original']
    
    for attention_type in attention_types:
        print(f"\n测试 {attention_type} 注意力:")
        
        model = EnhancedMHSIU(64, num_groups=4, attention_type=attention_type)
        
        with torch.no_grad():
            output = model(l, m, s)
            print(f"  输出形状: {output.shape}")
            print(f"  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试DSAM增强
    print(f"\n测试 DSAM 注意力:")
    model_dsam = DSAM_EnhancedMHSIU(64, num_groups=4)
    
    with torch.no_grad():
        output_dsam = model_dsam(l, m, s)
        print(f"  输出形状: {output_dsam.shape}")
        print(f"  参数量: {sum(p.numel() for p in model_dsam.parameters() if p.requires_grad):,}")
    
    print("\n=== MultiHeadMyNet 消融实验测试 ===")
    
    # 🔧 消融实验1: 头数消融
    print("\n1. 头数消融实验:")
    head_nums = [0, 1, 2, 6, 8]
    for num_heads in head_nums:
        model = MultiHeadMyNet(64, num_groups=3, num_heads=num_heads)
        with torch.no_grad():
            output = model(l, m, s)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  头数={num_heads}: 输出{output.shape}, 参数量={params:,}")
    
    # 🔧 消融实验2: 门控方式消融
    print("\n2. 门控方式消融实验:")
    gate_types = ['softmax', 'sigmoid', 'none']
    for gate_type in gate_types:
        model = MultiHeadMyNet(64, num_groups=3, num_heads=4, gate_type=gate_type)
        with torch.no_grad():
            output = model(l, m, s)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  门控={gate_type}: 输出{output.shape}, 参数量={params:,}")
    
    # 🔧 消融实验3: 头间融合方式消融
    print("\n3. 头间融合方式消融实验:")
    fusion_types = ['conv', 'weighted_sum']
    for fusion_type in fusion_types:
        model = MultiHeadMyNet(64, num_groups=3, num_heads=4, fusion_type=fusion_type)
        with torch.no_grad():
            output = model(l, m, s)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  融合={fusion_type}: 输出{output.shape}, 参数量={params:,}")
    
    # 🔧 消融实验4: 归一化与激活消融
    print("\n4. 归一化与激活消融实验:")
    norm_types = ['bn', 'gn']
    for norm_type in norm_types:
        model = MultiHeadMyNet(64, num_groups=3, num_heads=4, norm_type=norm_type)
        with torch.no_grad():
            output = model(l, m, s)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            activation = 'ReLU' if norm_type == 'bn' else 'SiLU'
            print(f"  归一化={norm_type.upper()}+{activation}: 输出{output.shape}, 参数量={params:,}")
    
    # 🔧 组合消融实验示例
    print("\n5. 组合消融实验示例:")
    ablation_configs = [
        {'num_heads': 4, 'gate_type': 'softmax', 'fusion_type': 'conv', 'norm_type': 'bn'},
        {'num_heads': 6, 'gate_type': 'sigmoid', 'fusion_type': 'weighted_sum', 'norm_type': 'gn'},
        {'num_heads': 2, 'gate_type': 'none', 'fusion_type': 'conv', 'norm_type': 'bn'},
    ]
    
    for i, config in enumerate(ablation_configs, 1):
        model = MultiHeadMyNet(64, num_groups=3, **config)
        with torch.no_grad():
            output = model(l, m, s)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  配置{i} {config}: 输出{output.shape}, 参数量={params:,}")
    
    print("\n🎉 所有测试完成！")
    print("📌 推荐使用 attention_type='scSE' 作为默认配置")
    print("📌 混合注意力 'hybrid' 可以获得更好的性能，但计算开销稍大")
    print("📌 MultiHeadMyNet 消融实验已添加，支持头数、门控、融合、归一化四个维度的对比")
