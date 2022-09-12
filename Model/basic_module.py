import torch
import torch.nn as nn
import functools
from Model.GDN_transform import GDN


# from .gdn import GDN
import numpy as np
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import BatchNorm2d as norm_layer
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']

class ResGDN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, inv=False):
        super(ResGDN, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.inv = bool(inv)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.ac1 = GDN(self.in_ch, self.inv)
        self.ac2 = GDN(self.in_ch, self.inv)

    def forward(self, x):
        x1 = self.ac1(self.conv1(x))
        x2 = self.conv2(x1)
        out = self.ac2(x + x2)
        return out


# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(ResBlock, self).__init__()
#         self.in_ch = int(in_channel)
#         self.out_ch = int(out_channel)
#
#         self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
#                                kernel_size=3, stride=1, padding=1)
#         self.leaky_relu=nn.LeakyReLU(inplace=True)
#         self.conv2 = nn.Conv2d(self.out_ch, self.out_ch,
#                                kernel_size=3,stride=1,padding=1)
#         if self.in_ch!=self.out_ch :
#             self.skip=nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1,stride=1)
#         else:
#             self.skip=None
#
#         # self.conv_block=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
#         #                               nn.BatchNorm2d(out_channel,0.8),
#         #                               nn.PReLU(),
#         #                               nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
#         #                               nn.BatchNorm2d(out_channel,0.8))
#
#     def forward(self, x):
#         identity=x
#         out=self.leaky_relu(self.conv1(x))
#         out=self.leaky_relu(self.conv2(out))
#         if self.skip is not None:
#             identity=self.skip(x)
#         out=identity+out
#         # out=x+self.conv_block(x)
#         return out



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               3,1,1)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch,
                               3,1,1)

    def forward(self, x):
        x1 = self.conv2(F.relu(self.conv1(x)))
        out = x+x1
        return out

class ResidualBlockWithStride(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3, stride=2):
        super(ResidualBlockWithStride,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.conv1=nn.Conv2d(self.in_ch,self.out_ch,self.k,stride=self.stride,padding=1)
        self.leaky_relu=nn.LeakyReLU(inplace=True)
        self.conv2=nn.Conv2d(self.out_ch,self.out_ch,self.k,stride=1,padding=1)
        self.gdn=GDN(self.out_ch)
        if self.in_ch!=self.out_ch or self.stride!=1 :
            self.skip=nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1,stride=self.stride)
        else:
            self.skip=None

    def forward(self,x):
        identity=x
        out=self.gdn(self.conv1(x))
        out=self.leaky_relu(self.conv2(out))
        if self.skip is not None:
            identity=self.skip(x)
        out=identity+out
        return out

class ResidualBlockUpsample(nn.Module):
    def __init__(self,in_channel, out_channel,upsample=2):
        super(ResidualBlockUpsample,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.subpel_conv=nn.Sequential(nn.Conv2d(self.in_ch,self.out_ch*upsample**2,kernel_size=3,padding=1),nn.PixelShuffle(upsample))
        self.leaky_relu=nn.LeakyReLU(inplace=True)
        self.conv=nn.Conv2d(self.out_ch,self.out_ch,kernel_size=3,stride=1,padding=1)
        self.igdn=GDN(self.out_ch,inverse=True)
        self.upsample=nn.Sequential(nn.Conv2d(self.in_ch,self.out_ch*upsample**2,kernel_size=3,padding=1),nn.PixelShuffle(upsample))

    def forward(self,x):
        identity=x
        out=self.igdn(self.subpel_conv(x))
        out=self.leaky_relu(self.conv(out))
        identity=self.upsample(x)
        out=out+identity
        return out





##

class BasicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1,groups=1,relu=True,bn=True,bias=False):
        super(BasicConv,self).__init__()
        self.out_channels=out_planes
        self.conv=nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.bn=nn.BatchNorm2d(out_planes,eps=1e-5,momentum=0.01,affine=True) if bn else None
        self.relu=nn.ReLU() if relu else None

    def forward(self,x):
        x=self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.relu is not None:
            x=self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),-1)

class ChannelGate(nn.Module):
    def __init__(self,gate_channels,reduction_ratio=16,pool_types=['avg','max']):
        super(ChannelGate,self).__init__()
        self.gate_channels=gate_channels
        self.mlp=nn.Sequential(Flatten(),
                               nn.Linear(gate_channels,gate_channels//reduction_ratio),
                               nn.ReLU(),
                               nn.Linear(gate_channels//reduction_ratio,gate_channels))
        self.pool_types=pool_types

    def forward(self,x):
        channel_att_sum=None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool=F.avg_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
                channel_att_raw=self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool=F.max_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
                channel_att_raw=self.mlp(max_pool)
            elif pool_type=='lp':
                lp_pool=F.lp_pool2d(x,2,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
                channel_att_raw=self.mlp(lp_pool)
            elif pool_type=='lse':
                lse_pool=logsumexp_2d(x)
                channel_att_raw=self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum=channel_att_raw
            else:
                channel_att_sum=channel_att_raw+channel_att_sum

            scale=F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
            return scale*x

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self,x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate,self).__init__()
        kernel_size=7
        self.compress=ChannelPool()
        self.spatial=BasicConv(2,1,kernel_size=kernel_size,stride=1,padding=(kernel_size-1)//2,relu=False)

    def forward(self,x):
        x_compress=self.compress(x)
        x_out=self.spatial(x_compress)
        scale=F.sigmoid(x_out)
        return scale*x

class CBAM(nn.Module):      ########## CBAM
    def __init__(self,gate_channels,reduction_ratio=16,pool_types=['avg','max'],no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate=ChannelGate(gate_channels,reduction_ratio,pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate=SpatialGate()
    def forward(self,x):
        x_out=self.ChannelGate(x)
        if not self.no_spatial:
            x_out=self.SpatialGate(x_out)
        return x_out


########################################################
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Attention(nn.Module):
    def __init__(self,N):
        super(Attention,self).__init__()
        self.main_enc_sa = PAM_Module(N//2)
        self.main_enc_sc = CAM_Module(N//2)

        self.conv5a = nn.Sequential(nn.Conv2d(N, N // 2, 3, padding=1, bias=False),
                                    norm_layer(N // 2),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(N, N // 2, 3, padding=1, bias=False),
                                    norm_layer(N // 2),
                                    nn.ReLU())

        self.conv51 = nn.Sequential(nn.Conv2d(N // 2, N // 2, 3, padding=1, bias=False),
                                    norm_layer(N // 2),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(N // 2, N // 2, 3, padding=1, bias=False),
                                    norm_layer(N // 2),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(N // 2, N, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(N // 2, N, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(N // 2, N, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.main_enc_sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.main_enc_sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        output.append(sa_conv)
        output.append(sc_conv)
        return tuple(output)


## Post-processing
def make_layer(block,n_layers):
    layers=[]
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self,nf=64,gc=32,bias=True):
        super(ResidualDenseBlock,self).__init__()
        self.conv1=nn.Conv2d(nf,gc,3,1,1,bias=bias)
        self.conv2=nn.Conv2d(nf+gc,gc,3,1,1,bias=bias)
        self.conv3=nn.Conv2d(nf+2*gc,gc,3,1,1,bias=bias)
        self.conv4=nn.Conv2d(nf+3*gc,gc,3,1,1,bias=bias)
        self.conv5=nn.Conv2d(nf+4*gc,nf,3,1,1,bias=bias)
        self.lrelu=nn.LeakyReLU(negative_slope=0.2,inplace=True)

    def forward(self,x):
        x1=self.lrelu(self.conv1(x))
        x2=self.lrelu(self.conv2(torch.cat((x,x1),1)))
        x3=self.lrelu(self.conv3(torch.cat((x,x1,x2),1)))
        x4=self.lrelu(self.conv4(torch.cat((x,x1,x2,x3),1)))
        x5=self.conv5(torch.cat((x,x1,x2,x3,x4),1))
        return x5*0.2+x

class RRDB(nn.Module):
    # Residual in Residual Dense Block
    def __init__(self,nf,gc=32):
        super(RRDB,self).__init__()
        self.RDB1=ResidualDenseBlock(nf,gc)
        self.RDB2=ResidualDenseBlock(nf,gc)
        # self.RDB3=ResidualDenseBlock(nf,gc)

    def forward(self,x):
        out=self.RDB1(x)
        out=self.RDB2(out)
        # out=self.RDB3(out)
        return out*0.2+x

class Post_Processing(nn.Module):
    def __init__(self,in_nc,out_nc,nf,nb,gc=32):
        super(Post_Processing,self).__init__()
        RRDB_block_f=functools.partial(RRDB,nf=nf,gc=gc)
        self.conv_first=nn.Conv2d(in_nc,nf,3,1,1,bias=True)
        self.RRDB_trunk=make_layer(RRDB_block_f,nb)
        self.trunk_conv=nn.Conv2d(nf,nf,3,1,1,bias=True)
        self.conv=nn.Sequential(nn.Conv2d(nf,nf,3,1,1,bias=True),
                      nn.LeakyReLU(negative_slope=0.2,inplace=True),
                      nn.Conv2d(nf,nf,3,1,1,bias=True),
                      nn.LeakyReLU(negative_slope=0.2,inplace=True),
                      nn.Conv2d(nf,nf,3,1,1,bias=True),
                      nn.LeakyReLU(negative_slope=0.2,inplace=True),
                      nn.Conv2d(nf,out_nc,3,1,1,bias=True))
        # self.conv = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        #                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                           nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True))

    def forward(self,x):
        fea=self.conv_first(x)
        trunk=self.trunk_conv(self.RRDB_trunk(fea))
        fea=fea+trunk
        out=self.conv(fea)
        return out

###Post-processing
class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        # self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        a0=self.b1(inputs)
        c0=torch.cat([inputs,a0],1)
        a1=self.b2(c0)
        c1=torch.cat([c0,a1],1)
        a2=self.b3(c1)
        c2=torch.cat([c1,a2],1)
        a3=self.b4(c2)
        c3=torch.cat([c2,a3],1)
        out=self.b5(c3)
        # for block in self.blocks:
        #     out = block(inputs)
        #     inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters)
        ) #, DenseResidualBlock(filters)

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
#
#
# class GeneratorRRDB(nn.Module):
#     def __init__(self, channels, filters=64, num_res_blocks=16):
#         super(GeneratorRRDB, self).__init__()
#
#         # First layer
#         self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
#         # Residual blocks
#         self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
#         # Second conv layer post residual blocks
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#         # Upsampling layers
#         # upsample_layers = []
#         # for _ in range(num_upsample):
#         #     upsample_layers += [
#         #         nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
#         #         nn.LeakyReLU(),
#         #         nn.PixelShuffle(upscale_factor=2),
#         #     ]
#         # self.upsampling = nn.Sequential(*upsample_layers)
#         # Final output block
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
#         )
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         # out = self.upsampling(out)
#         out = self.conv3(out)
#         return out



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)

    def forward(self, x):
        x1 = self.conv2(F.relu(self.conv1(x)))
        out = x+x1
        return out

# here use embedded gaussian


class Non_local_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Non_local_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.out_channel, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f1, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z