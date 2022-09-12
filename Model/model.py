import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.uniform import Uniform
from Model.context_model import P_Model
from Model.factorized_entropy_model import Entropy_bottleneck
from Model.gaussian_entropy_model import Distribution_for_entropy
from normalisation import channel, instance
from torchvision.models import vgg19
from Model.super_resolution import MeanShift
from Model.GDN_transform import GDN
from Model.GCN import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
from Model.super_resolution import ACBlock
from Model.Coord_Attention import CoordAtt

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.feature_extractor(img)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

## Channel Attention (CA, part of encoder/decoder)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB, part of encoder/decoder)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction=1,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG, part of encoder/decoder)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, n_blocks, kernel_size, reduction=1, act=nn.ReLU(True), res_scale=1):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_blocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## default convolution (part of encoder/decoder)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, dim=1)
    r_inv = 1 / (rowsum + 1e-10)
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx

class Enc(nn.Module):
    def __init__(self, num_features, M,N2,activation='relu', channel_norm=True):
        #input_features = 3, M=192, N1 = 320, N2=128
        super(Enc, self).__init__()

        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.M=M
        self.N2 = N2
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.beta1= nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.beta2 = nn.Parameter(torch.ones(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.beta3 = nn.Parameter(torch.ones(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.beta4 = nn.Parameter(torch.ones(1))
        self.gamma5 = nn.Parameter(torch.zeros(1))
        self.beta5 = nn.Parameter(torch.ones(1))
        self.sub_mean = MeanShift(rgb_range=1, sign=-1)


        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(2)
        self.asymmetric_pad = nn.ReflectionPad2d((0, 1, 1, 0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(num_features, N2//2, kernel_size=5, stride=1),
            self.interlayer_norm(N2//2, **norm_kwargs),
            self.activation(),
        )
        self.blk1 = ACBlock(N2 // 2, N2 // 2)
        self.low_high = nn.Conv2d(N2 // 2, M, 5, 16,2)
        self.trans=nn.Conv2d(N2,M,3,1,1)
        encode_modules_body1 = []
        encode_modules_body1.append(ResidualGroup(default_conv, N2, 6, 3, act=nn.ReLU(True)))
        self.encode1=nn.Sequential(*encode_modules_body1)

        encode_modules_body2 = []
        encode_modules_body2.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.encode2 = nn.Sequential(*encode_modules_body2)

        encode_modules_body3 = []
        encode_modules_body3.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.encode3 = nn.Sequential(*encode_modules_body3)

        encode_modules_body4 = []
        encode_modules_body4.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.encode4 = nn.Sequential(*encode_modules_body4)

        for idx in range(1, 4):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(N2 // 2, N2 // 2)))

        self.down1 = nn.Sequential(nn.Conv2d(N2 // 2, N2, 5, 2, 2), GDN(N2))

        for idx in range(4, 7):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(N2, N2)))

        self.down2 = nn.Sequential(nn.Conv2d(N2, M, 5, 2, 2), GDN(M))

        for idx in range(7, 10):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))

        self.down3 = nn.Sequential(nn.Conv2d(M, M, 5, 2, 2), GDN(M))
        self.attention1=CoordAtt(N2,N2)
        self.attention2=CoordAtt(M,M)
        self.attention3=CoordAtt(M,M)
        self.attention4=CoordAtt(M,M)

        for idx in range(10, 13):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))

        self.down4 = nn.Sequential(nn.Conv2d(M, M, 5, 2, 2), GDN(M))

        self.conv_out = nn.Conv2d(M, N2, 3, 1, 1)

        self.ave_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # hyper
        self.gcn1 = GraphAttentionLayer(M, M, 0.6, 0.2)
        self.gcn2 = GraphAttentionLayer(M, M, 0.6, 0.2)
        hyper_encode_modules_body1= []
        hyper_encode_modules_body1.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.hyper_encode1 = nn.Sequential(*hyper_encode_modules_body1)

        hyper_encode_modules_body2 = []
        hyper_encode_modules_body2.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.hyper_encode2 = nn.Sequential(*hyper_encode_modules_body2)


        for idx in range(13, 16):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))
        self.down5 = nn.Sequential(nn.Conv2d(M, M, 5, 2, 2),nn.LeakyReLU())

        for idx in range(16, 19):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))
        self.down6 = nn.Sequential(nn.Conv2d(M, M, 5, 2, 2),nn.LeakyReLU())

        self.attention5=CoordAtt(M,M)
        self.attention6=CoordAtt(M,M)

        self.conv=nn.Conv2d(M,N2,3,1,1)

    def forward(self, x):
        x0=self.sub_mean(x)
        x0=self.conv_block1(x0)
        x1 = self.blk1(x0)
        high1, tmp = x1, x1
        for idx in range(1, 4):
            tmp = self.__getattr__(f"blk{idx + 1}")(tmp)
            high1 = high1 + tmp
        tmp_1 = self.down1(high1)
        encode1_channel=self.encode1(tmp_1)
        tmp_1=self.attention1(tmp_1)+encode1_channel
        rec_attention=tmp_1
        identity_at1=tmp_1
        high2,tmp1 = tmp_1,tmp_1
        for idx in range(4, 7):
            tmp1 = self.__getattr__(f"blk{idx + 1}")(tmp1)
            high2 = high2 + tmp1
        tmp_2 = self.down2(high2)
        encode2_channel = self.encode2(tmp_2)
        tmp_2=self.attention2(tmp_2)+encode2_channel
        tmp_2=self.beta1*tmp_2+self.gamma1*self.ave_pool(self.trans(identity_at1))
        identity_at2=tmp_2
        high3,tmp2 = tmp_2,tmp_2
        for idx in range(7, 10):
            tmp2 = self.__getattr__(f"blk{idx + 1}")(tmp2)
            high3 = high3 + tmp2
        tmp_3 = self.down3(high3)
        encode3_channel=self.encode3(tmp_3)
        tmp_3 =self.attention3(tmp_3)+encode3_channel
        tmp_3=self.gamma2*self.ave_pool(identity_at2)+self.beta2*tmp_3
        identity_at3=tmp_3

        high4_1, at_map =tmp_3,tmp_3

        for idx in range(10, 13):
            at_map = self.__getattr__(f"blk{idx + 1}")(at_map)
            high4_1 = high4_1 + at_map


        # high5 = self.low_high(x1) + high4_1  ########output1 fusion
        high5 = self.down4(high4_1)
        high5=self.low_high(x1) + high5
        encode4_channle=self.encode4(high5)
        output1=self.attention4(high5)+encode4_channle
        output1=self.beta3*output1+self.gamma3*self.ave_pool(identity_at3)

        # hyper
        high6, tmp6 = output1, output1
        identity_at4 = output1
        for idx in range(13, 16):
            tmp6 = self.__getattr__(f"blk{idx + 1}")(tmp6)
            high6 = high6 + tmp6
        hyper1=self.down5(high6)
        hyper1_channel=self.hyper_encode1(hyper1)
        identity = hyper1
        hyper1 = self.attention5(hyper1)+hyper1_channel
        hyper1=self.gamma4*self.ave_pool(identity_at4)+self.beta4*hyper1
        identity_at5=hyper1

        for i in range(identity.shape[0]):
            feat=identity[i].permute(1,2,0).reshape(-1,identity.shape[1])
            adj1=torch.mm(feat,torch.t(feat))
            adj1=row_normalize(adj1)
            gc=f.relu(self.gcn1(feat,adj1))
            gc=f.relu(self.gcn2(gc,adj1))
            gc=gc.reshape(identity[i].shape[1],identity[i].shape[2],self.M)
            gc=gc.permute(2,0,1)
            gc=gc.unsqueeze(0)
            if i==0:
                gc_out=gc
            else:
                gc_out=torch.cat((gc_out,gc),dim=0)

        hyper1=hyper1+gc_out


        high7, tmp7 =hyper1,hyper1
        for idx in range(16, 19):
            tmp7 = self.__getattr__(f"blk{idx + 1}")(tmp7)
            high7 = high7 + tmp7
        x8=self.down6(high7)
        hyper2_channel=self.hyper_encode2(x8)
        x9 = self.beta5*(self.attention6(x8)+hyper2_channel)+self.gamma5*self.ave_pool(identity_at5)

        output2=self.conv(x9)
        return output1, output2


class Hyper_Dec(nn.Module):
    def __init__(self, M,N2):
        super(Hyper_Dec, self).__init__()

        self.conv0=nn.Conv2d(N2,M,kernel_size=3,stride=1,padding=1)

        self.attention1=CoordAtt(M,M)
        self.attention2 = CoordAtt(M, M)
        self.M=M
        self.N2 = N2
        self.gcn1 =GraphAttentionLayer(M, M,0.6,0.2)
        self.gcn2 =GraphAttentionLayer(M, M,0.6,0.2)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.subpel_conv1 = nn.Sequential(nn.Conv2d(M, M * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        hyper_decode_modules_body1 = []
        hyper_decode_modules_body1.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.hyper_decode1 = nn.Sequential(*hyper_decode_modules_body1)

        hyper_decode_modules_body2 = []
        hyper_decode_modules_body2.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.hyper_decode2 = nn.Sequential(*hyper_decode_modules_body2)

        for idx in range(1, 4):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))
        self.up1=nn.Sequential(nn.ConvTranspose2d(M,M,5,2,2,1),nn.LeakyReLU())

        for idx in range(4, 7):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))
        self.up2=nn.Sequential(nn.ConvTranspose2d(M,M,5,2,2,1),nn.LeakyReLU())

    def forward(self, x):
        x0=self.conv0(x)
        x1=self.attention1(x0)+self.hyper_decode1(x0)

        x3 = self.up1(x1)
        high1,tmp=x3,x3
        for idx in range(1, 4):
            tmp = self.__getattr__(f"blk{idx + 1}")(tmp)
            high1 = high1 + tmp
        identity=high1
        hyper2_channel=self.hyper_decode2(high1)

        for i in range(high1.shape[0]):
            feat=high1[i].permute(1,2,0).reshape(-1,high1.shape[1])
            adj1=torch.mm(feat,torch.t(feat))
            adj1=row_normalize(adj1)
            gc=f.relu(self.gcn1(feat,adj1))
            gc=f.relu(self.gcn2(gc,adj1))
            gc=gc.reshape(high1[i].shape[1],high1[i].shape[2],self.M)
            gc=gc.permute(2,0,1)
            gc=gc.unsqueeze(0)
            if i==0:
                gc_out=gc
            else:
                gc_out=torch.cat((gc_out,gc),dim=0)

        high1=self.beta1*(self.attention2(identity)+hyper2_channel)+self.gamma1*self.subpel_conv1(x1)
        high1=high1+gc_out

        x4=self.up2(high1)

        high2, tmp1 = x4, x4
        for idx in range(4, 7):
            tmp1 = self.__getattr__(f"blk{idx + 1}")(tmp1)
            high2 = high2 + tmp1

        return high2

class Dec(nn.Module):
    def __init__(self, M,N2, input): #,activation='relu', channel_norm=True
        super(Dec, self).__init__()
        self.M=M

        self.post_pad = nn.ReflectionPad2d(2)
        self.subpel_conv1 = nn.Sequential(nn.Conv2d(M, M * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.subpel_conv2 = nn.Sequential(nn.Conv2d(M, M * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))
        self.subpel_conv3 = nn.Sequential(nn.Conv2d(M, N2 * 2 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(2))

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.beta2 = nn.Parameter(torch.ones(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.beta3 = nn.Parameter(torch.ones(1))
        self.add_mena = MeanShift(rgb_range=1, sign=1)

        self.attention1=CoordAtt(M,M)
        self.attention3=CoordAtt(M,M)
        self.attention2=CoordAtt(M,M)
        self.attention4=CoordAtt(N2,N2)

        decode_modules_body1 = []
        decode_modules_body1.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.decode1 = nn.Sequential(*decode_modules_body1)

        decode_modules_body2 = []
        decode_modules_body2.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.decode2 = nn.Sequential(*decode_modules_body2)

        decode_modules_body3 = []
        decode_modules_body3.append(ResidualGroup(default_conv, M, 6, 3, act=nn.ReLU(True)))
        self.decode3 = nn.Sequential(*decode_modules_body3)

        decode_modules_body4 = []
        decode_modules_body4.append(ResidualGroup(default_conv, N2, 6, 3, act=nn.ReLU(True)))
        self.decode4 = nn.Sequential(*decode_modules_body4)

        for idx in range(1, 4):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(M, M, 5, 2, 2, 1), GDN(M, inverse=True))

        for idx in range(4, 7):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(M, M)))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(M, M, 5, 2, 2, 1),GDN(M,inverse=True))

        for idx in range(7, 10):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(N2, N2)))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(M, N2, 5, 2, 2, 1),GDN(N2,inverse=True))

        for idx in range(10, 13):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(N2//2, N2//2)))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(N2, N2 // 2, 5, 2, 2, 1),GDN(N2//2,inverse=True))


        self.blk1=ACBlock(N2//2,N2//2)
        self.high_low=nn.ConvTranspose2d(M,N2//2,5,16,2,15)

        self.conv=nn.Sequential(
            self.post_pad,
            nn.Conv2d(N2//2, input, kernel_size=5, stride=1),
        )

    def forward(self, x):
        x0=self.attention1(x)+self.decode1(x)
        x3 = self.up1(x0)

        high1,tmp1=x3,x3
        for idx in range(1, 4):
            tmp1 = self.__getattr__(f"blk{idx + 1}")(tmp1)
            high1 = high1 + tmp1

        high1=self.attention2(high1)+self.decode2(high1)
        high1=self.beta1*high1+self.gamma1*self.subpel_conv1(x0)
        x4=self.up2(high1)

        high2, tmp2 = x4, x4
        for idx in range(4, 7):
            tmp2 = self.__getattr__(f"blk{idx + 1}")(tmp2)
            high2 = high2 + tmp2

        x5=self.attention3(high2)+self.decode3(high2)
        x5=self.beta2*x5+self.gamma2*self.subpel_conv2(high1)
        x6=self.up3(x5)
        high3, tmp3 = x6, x6
        for idx in range(7, 10):
            tmp3 = self.__getattr__(f"blk{idx + 1}")(tmp3)
            high3 = high3 + tmp3

        high3=self.attention4(high3)+self.decode4(high3)
        high3=self.beta3*high3+self.gamma3*self.subpel_conv3(x5)

        x7 = self.up4(high3)
        high4,tmp4=x7,x7
        for idx in range(10, 13):
            tmp4= self.__getattr__(f"blk{idx + 1}")(tmp4)
            high4 = high4 + tmp4

        x8 = high4 + self.high_low(x0)
        x9=self.blk1(x8)
        output=self.conv(x9)
        output=self.add_mena(output)

        return output


class Image_coding(nn.Module):
    def __init__(self, input_features,M,N2): #,nf,nb,gc=32
        #input_features = 3, M = 192, N = 128
        super(Image_coding, self).__init__()
        self.encoder = Enc(input_features, M, N2,activation='relu', channel_norm=True)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(M, N2)
        self.p = P_Model(M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.decoder = Dec(M,N2,input_features)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, if_training=1):
        x1, x2 = self.encoder(x)
        xq2, xp2 = self.factorized_entropy_func(x2, if_training)
        x3= self.hyper_dec(xq2)
        hyper_dec = self.p(x3)
        if if_training == 0:
            xq1 = self.add_noise(x1)
        elif if_training == 1:
            xq1 = UniverseQuant.apply(x1)
        else:
            xq1 = torch.round(x1)
        xp1 = self.gaussin_entropy_func(xq1, hyper_dec)

        output = self.decoder(xq1)

        return [output, xp1, xp2, xq1, hyper_dec]


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1, 1)
        #b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g

if __name__=='__main__':
    input=torch.randn((12,3,256,256)).cuda()
    net=Image_coding(3,192,128).cuda()
    a=net(input)

    print(a.size())