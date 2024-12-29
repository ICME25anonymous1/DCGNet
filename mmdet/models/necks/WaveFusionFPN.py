import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule, auto_fp16
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from ..builder import NECKS
from ..utils.convolution import Conv
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import os

def draw_feature_map1(features, img_path='demo.jpg', save_dir = 'feature_map/refine',name = '_'):
    '''
    :param features: 特征层。可以是单层，也可以是一个多层的列表
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:png
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        heatmap = featuremap_2_heatmap1(featuremap)	#主要是这个，就是取特征层整个的求和然后平均，归一化
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.6 + img  # 这里的0.4是热力图强度因子
        matplotlib.use('Agg')
        plt.imshow(heatmap)  # ,cmap='gray' ，这里展示下可视化的像素值
        # plt.imshow(superimposed_img)  # ,cmap='gray'
        plt.savefig('out.jpg')
        plt.close()	#关掉展示的图片
        # 下面是用opencv查看图片的
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(os.path.join(save_dir, name + str(i) + '.png'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看

        print(os.path.join(save_dir, name + str(i) + '.png'))
        i = i + 1
def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = Ours feature map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)
    return heatmap

class DynamicScale(BaseModule):
    def __init__(self, input_dim=256, head_num=4, out_channels=256):
        super(DynamicScale, self).__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        one_head_channels = input_dim // head_num
        self.up_channel = nn.ModuleList([nn.Conv2d(one_head_channels, input_dim, 1, 1) for _ in range(head_num)])
        # self.up_channel = nn.ModuleList([
        #     nn.Conv2d(51, input_dim, 1, 1),
        #     nn.Conv2d(51, input_dim, 1, 1),
        #     nn.Conv2d(51, input_dim, 1, 1),
        #     nn.Conv2d(51, input_dim, 1, 1),
        #     nn.Conv2d(52, input_dim, 1, 1)
        # ])
        self.block = self.create_block(head_num, input_dim)
        self.mix1 = nn.Conv2d(input_dim * head_num, input_dim * head_num, 1, 1, groups=input_dim)
        self.bn = nn.BatchNorm2d(input_dim * head_num)
        self.act = nn.GELU()
        self.mix2 = nn.Conv2d(input_dim * head_num, out_channels, 1, 1)

    def create_block(self, head_num, input_dim):
        block_list = []
        for i in range(head_num):
            kernel = 3 + i * 2
            block = nn.Sequential(
                nn.Conv2d(input_dim, input_dim // 2, 1),
                DepthwiseSeparableConvModule(input_dim // 2, input_dim // 2, kernel, 1, kernel//2, 1),
                nn.BatchNorm2d(input_dim // 2),
                nn.LeakyReLU(),
                nn.Conv2d(input_dim // 2, input_dim, 1)
            )
            block_list.append(block)
        return nn.ModuleList(block_list)

    @auto_fp16()
    def forward_feature(self, x, B, H, W):
        tmp_feature = []
        for i in range(self.head_num):
            block = self.block[i]
            cur_feature = self.up_channel[i](x[i])
            if len(tmp_feature) != 0:
                cur_feature = cur_feature + tmp_feature[-1]
            tmp = block(cur_feature)
            cur_feature = self.act(cur_feature + tmp)
            tmp_feature.append(cur_feature)
        out_feature = torch.stack(tmp_feature, dim=2).contiguous().view(B, -1, H, W)
        context = self.mix2(self.act(self.mix1(out_feature)))
        return context

    @auto_fp16()
    def forward(self, x, B, H, W):
        context = self.forward_feature(x, B, H, W)
        return context


class DynamicScaleVersion2(BaseModule):
    def __init__(self, input_dim=256, head_num=2, out_channels=256):
        super(DynamicScaleVersion2, self).__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        one_head_channels = input_dim // head_num
        self.block = self.create_block(head_num, one_head_channels)
        self.mix1 = nn.Conv2d(input_dim, input_dim, 1, 1, groups=input_dim)
        self.bn = nn.BatchNorm2d(input_dim * head_num)
        self.act = nn.GELU()
        self.mix2 = nn.Conv2d(input_dim, input_dim, 1, 1)

    def create_block(self, head_num, input_dim):
        block_list = []
        for i in range(head_num):
            kernel = 3 + i * 2
            block = nn.Sequential(
                DepthwiseSeparableConvModule(input_dim, input_dim, kernel, 1, kernel//2, 1),
                nn.BatchNorm2d(input_dim),
                nn.LeakyReLU(),
            )
            block_list.append(block)
        return nn.ModuleList(block_list)

    @auto_fp16()
    def forward_feature(self, x, B, H, W):
        tmp_feature = []
        #print(len(x))
        for i in range(self.head_num):
            block = self.block[i]
            cur_feature = x[i]
            if len(tmp_feature) != 0:
                cur_feature = cur_feature + tmp_feature[-1]
            tmp = block(cur_feature)
            cur_feature = cur_feature + tmp
            tmp_feature.append(cur_feature)
        out_feature = torch.stack(tmp_feature, dim=2).contiguous().view(B, -1, H, W)
        context = self.mix2(self.act(self.mix1(out_feature)))
        return context

    @auto_fp16()
    def forward(self, x, B, H, W):
        context = self.forward_feature(x, B, H, W)
        return context

class Sobel2D(BaseModule):
    def __init__(self, in_channels=256):
        super(Sobel2D, self).__init__()
        self.x_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.y_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        x_sobel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).expand(in_channels, in_channels, -1, -1).clone()
        y_sobel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).expand(in_channels, in_channels, -1, -1).clone()
        self.x_conv.weight = nn.Parameter(x_sobel)
        self.y_conv.weight = nn.Parameter(y_sobel)
        self.act = nn.ReLU()

    @auto_fp16()
    def forward(self, x):
        x_edge = self.act(self.x_conv(x))
        y_edge = self.act(self.y_conv(x))
        texture = torch.sqrt(x_edge**2 + y_edge**2)
        # out = x + texture
        # return out
        return texture

class NoiseRemoveAttention(BaseModule):
    def __init__(self, in_channels=1024):
        super(NoiseRemoveAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.mix = nn.Conv1d(1, 1, 3, 1, 1)
        # self.act1 = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 7, 1, 3)
        self.act2 = nn.Sigmoid()
        self.gConvMix = nn.Conv2d(in_channels * 2, in_channels * 2, 7, 1, 3, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels//4, 1, 1, 0)
        self.act3 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels//4, in_channels, 1, 1, 0)
        self.conv4 = nn.Conv2d(in_channels * 2, in_channels, 1, 1)
        self.act4 = nn.Sigmoid()

    def forward(self, high_frequency):
        B, _, H, W = high_frequency.shape
        max_feature = torch.max(high_frequency, dim=1, keepdim=True)[0]
        avg_feature = torch.mean(high_frequency, dim=1, keepdim=True)
        cat_avg_mean = torch.cat([max_feature, avg_feature], dim=1)
        spatial_att = self.act2(self.conv1(cat_avg_mean))
        avg_channel_feature = self.pool(high_frequency)
        channel_att = self.conv3(self.act3(self.conv2(avg_channel_feature)))
        coarse_att = spatial_att + channel_att
        cat_att_content = torch.stack([coarse_att, high_frequency], dim=2).view(B, -1, H, W)
        refine_att = self.conv4(self.act4(self.gConvMix(cat_att_content)))
        return high_frequency + high_frequency * refine_att

class RefineAttention(BaseModule):
    def __init__(self, in_channel=0):
        super(RefineAttention, self).__init__()
        assert in_channel > 0
        self.in_channel = in_channel
        self.stem1 = Conv(in_channel, in_channel, 1, 0, 1, relu=False)
        self.stem2 = Conv(in_channel, in_channel, 3, 1, 1)
        self.stem3 = Conv(in_channel, in_channel, 3, 1, 1)
        self.mix1 = Conv(in_channel, in_channel, 3, 1, 1)
        self.mix2 = nn.Conv2d(in_channel * 2, in_channel * 2, 1, 1, 0, groups=2)
        self.act = nn.ReLU()
        self.mix3 = nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0, groups=2)

    def forward_feature(self, x):
        B, C, H, W = x.shape
        assert self.in_channel > 0
        fea1 = self.stem1(x)
        fea2 = self.stem2(x)
        fea3 = self.stem3(x)
        horizontal_max = torch.max(x, dim=3, keepdim=True)[0]
        vertical_max = torch.max(x, dim=2, keepdim=True)[0]
        sum_fea = self.mix1(horizontal_max + vertical_max)
        stack_fea = torch.stack([sum_fea, fea1], dim=2).view(B, -1, H, W)
        att = self.mix3(self.act(self.mix2(stack_fea)))
        return att * x + x

    def forward(self, x):
        out = self.forward_feature(x)
        return out

class HighFrequencyRefine(nn.Module):
    def __init__(self, inchannels=256):
        super(HighFrequencyRefine, self).__init__()
        self.conv1 = ConvModule(inchannels * 2, inchannels, 3, 1, 1)
        self.conv2 = ConvModule(inchannels * 2, inchannels, 3, 1, 1)
        self.conv3 = ConvModule(inchannels * 2, inchannels, 3, 1, 1)
        self.conv4 = nn.Conv2d(2, 3, 5, 1, 2)
        #self.norm = nn.BatchNorm2d(inchannels * 3)
        #self.conv2 = DepthwiseSeparableConvModule(inchannels * 2, inchannels * 3, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, LL, LH, HL, HH):
        LL_LH = torch.cat([LL, LH], dim=1)
        LL_HL = torch.cat([LL, HL], dim=1)
        LL_HH = torch.cat([LL, HH], dim=1)
        refine_LH = self.conv1(LL_LH)
        refine_HL = self.conv2(LL_HL)
        refine_HH = self.conv3(LL_HH)
        sum_fea = refine_LH + refine_HL + refine_HH
        max_fea = torch.max(sum_fea, dim=1, keepdim=True)[0]
        mean_fea = torch.mean(sum_fea, dim=1, keepdim=True)
        cat_fea =torch.cat([max_fea, mean_fea], dim=1)
        att = self.act(self.conv4(cat_fea))
        LH_att, HL_att, HH_att = torch.chunk(att, 3, 1)
        LH_out, HL_out, HH_out = LH + LH_att * refine_LH, HL + HL_att * refine_HL, HH + HH_att * refine_HH
        return LH_out, HL_out, HH_out

class WaveRefineBlock(BaseModule):
    def __init__(self, in_channels=None):
        super(WaveRefineBlock, self).__init__()
        self.sobel = Sobel2D(in_channels)
        self.dwt_2D = DWT_2D('haar')
        self.idwt_2D = IDWT_2D('haar')
        self.att = nn.ModuleList([NoiseRemoveAttention(in_channels) for _ in range(3)])
        self.high_frequency_refine = HighFrequencyRefine(256)

    def forward(self, x):
        x = self.sobel(x)
        LL, LH, HL, HH = self.dwt_2D(x)
        high_f = [LH, HL, HH]
        for index, item in enumerate(high_f):
            out = self.att[index](item)
            if index == 0:
                LH = out
            elif index == 1:
                HL = out
            else:
                HH = out
        #LH, HL, HH = self.high_frequency_refine(LL, LH, HL, HH)
        out = self.idwt_2D(LL, LH, HL, HH)
        return out

class DWTRefineBlock(BaseModule):
    def __init__(self, in_channels=None, head_num=2):
        super(DWTRefineBlock, self).__init__()
        self.head_num = head_num
        #self.adaptive_scale = DynamicScale()
        self.adaptive_scale = DynamicScaleVersion2()
        self.wave_refine = WaveRefineBlock(in_channels=in_channels)

    @auto_fp16()
    def forward_feature(self, x):
        out = self.wave_refine(x)
        B, C, H, W = out.shape
        #cur_feature = [out.clone()[:, 0:51,...], out.clone()[:, 51:102,...], out.clone()[:,102:153,...], out.clone()[:, 153:204, ...], out.clone()[:, 204:, ...]]
        cur_feature = out.clone().permute(0, 2, 3, 1).contiguous().view(B, H, W, self.head_num, -1).permute(
            3, 0, 4, 1, 2)
        out = out + self.adaptive_scale(cur_feature, B, H, W)
        return out

    def forward(self, x):
        out = self.forward_feature(x)
        return out

class AdaptiveFusion(BaseModule):
    def __init__(self, fea_num=2, input_dim=256):
        super(AdaptiveFusion, self).__init__()
        self.input_dim = input_dim
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mix = nn.Conv2d(input_dim * 4, input_dim * 2, 1)
        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_dim * 2, 2, 1)

    @auto_fp16()
    def forward_feature(self, x1, x2):
        cat_fea = torch.cat([x1, x2], dim=1)
        max_fea = self.max_pool(cat_fea)
        avg_fea = self.avg_pool(cat_fea)
        tmp = torch.cat([max_fea, avg_fea], dim=1)
        tmp = self.act(self.mix(tmp))
        spatial_fea = self.conv1(cat_fea)
        spatial_fea = F.softmax(spatial_fea, dim=1)
        out = x1 * tmp[:, :self.input_dim, ...] * spatial_fea[:, 0, ...].unsqueeze(1) + x2 * tmp[:, self.input_dim:, ...] * spatial_fea[:, 1, ...].unsqueeze(1)
        return out
        # x = torch.cat([x1, x2], dim=1)
        # mean_feature = torch.mean(x, dim=1, keepdim=True)
        # max_feature = torch.max(x, dim=1, keepdim=True)[0]
        # cat_feature = torch.cat([mean_feature, max_feature], dim=1)
        # mix_feature = self.act(self.mix(cat_feature))
        # out = mix_feature[:, 0, ...].unsqueeze(1) * x1 + mix_feature[:, 1, ...].unsqueeze(1) * x2
        # return out

    def forward(self, x1, x2):
        out = self.forward_feature(x1, x2)
        return out

@NECKS.register_module()
class WaveFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(WaveFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.feature_refine = nn.ModuleList([DWTRefineBlock(out_channels) for _ in range(3 + 1)])
        self.adaptive_fusion = nn.ModuleList([AdaptiveFusion() for _ in range(3)])
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward_feature(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                upsample_feature = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                #laterals[i - 1] += upsample_feature
                #wavelets对上层特征进行精炼
                refined_feature = self.feature_refine[i - 1](upsample_feature)
                #refined_feature = upsample_feature
                laterals[i - 1] += refined_feature
                #自适应的融合
                #laterals[i - 1] = self.adaptive_fusion[i - 1](laterals[i - 1], upsample_feature)
                # laterals[i - 1] += F.interpolate(
                #     laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        #matplotlib.use('Agg')
        #draw_feature_map1(laterals[0])
        # tmp_tensor = laterals[0][0, 1, ...]
        # tmp_tensor = (tmp_tensor - tmp_tensor.min()) / (tmp_tensor.max() - tmp_tensor.min()) * 255
        # tmp_np = tmp_tensor.detach().cpu().numpy()
        # plt.imshow(tmp_np)
        # plt.axis('off')
        # plt.title('test')
        # plt.savefig('out.jpg')
        laterals[0] = self.feature_refine[-1](laterals[0])
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    @auto_fp16()
    def forward_adaptiveScale(self, inputs):
        inputs_list = list(inputs)
        out = self.forward_feature(inputs)
        out_list = list(out)
        if self.context_adaptive0 and self.context_adaptive1:
            for i in range(2):
                B, C, H, W = inputs_list[i].shape
                cur_feature = inputs_list[i].permute(0, 2, 3, 1).contiguous().view(B, H, W, 4, -1).permute(
                    3, 0, 4, 1, 2)
                context_adaptive = getattr(self, f'context_adaptive{i}')
                context = context_adaptive(cur_feature, B, H, W)
                # fusion
                out_list[i] = out_list[i] + context
        return tuple(out_list)

    @auto_fp16()
    def forward(self, inputs):
        # out = self.forward_adaptiveScale(inputs)
        # return out
        out = self.forward_feature(inputs)
        return out
