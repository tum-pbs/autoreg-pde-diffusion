import torch
import torch.nn as nn
import torch.nn.functional as F

from turbpred.params import DataParams, ModelParamsDecoder, ModelParamsEncoder, ModelParamsLatent

# latent space input shape: B S L
def reparameterizeAndSampleLatentSpaceVAE(latentSpace:torch.Tensor, simParam:torch.Tensor):
    if latentSpace.shape[2] % 2 != 0:
        raise ValueError("Uneven number of latent variables for VAE!")
    latentSize = int(0.5 * latentSpace.shape[2])
    vaeMean = latentSpace[:, :, 0:latentSize]
    vaeLogVar = latentSpace[:, :, latentSize:latentSpace.shape[2]]

    std = torch.exp(0.5 * vaeLogVar)
    #epsilon = torch.randn_like(std)
    epsilon = torch.randn((std.shape[0], 1, std.shape[2]), device="cuda" if std.is_cuda else "cpu").expand(-1,std.shape[1],-1)
    result = vaeMean + (epsilon * std)
    if simParam is not None:
        result[:, :, 0:simParam.shape[2]] = simParam # overwrite sampling for simulation parameters
    return result, vaeMean, vaeLogVar




class EncoderModelSkip(nn.Module):
    p_me: ModelParamsEncoder
    p_ml: ModelParamsLatent

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder, p_ml:ModelParamsLatent, dimension:int):
        super(EncoderModelSkip, self).__init__()

        self.p_d = p_d
        self.p_me = p_me
        self.p_ml = p_ml
        self.dimension = dimension

        eW = self.p_me.encWidth
        lat = self.p_me.latentSize
        inChannel = dimension + len(self.p_d.simFields) + len(self.p_d.simParams)

        if self.dimension == 2:
            self.encPoolSkip1 = nn.AvgPool2d(kernel_size=8, stride=8)
            self.encPoolSkip2 = nn.AvgPool2d(kernel_size=16, stride=16)

            self.encConv1 = nn.Sequential(
                nn.Conv2d(inChannel, eW, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.encConv2 = nn.Sequential(
                nn.Conv2d(eW + inChannel, 3*eW, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.encConv3 = nn.Sequential(
                nn.Conv2d(3*eW + inChannel, 6*eW, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.encConv4 = nn.Sequential(
                nn.Conv2d(6*eW + inChannel, 4*eW, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.encConv5 = nn.Sequential(
                nn.Conv2d(4*eW + inChannel, eW, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.encConv6 = nn.Sequential(
                nn.Conv2d(eW + inChannel, lat, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.encPool = nn.AdaptiveAvgPool2d(1)

        elif self.dimension == 3:
            self.encPoolSkip1 = nn.AvgPool3d(kernel_size=8, stride=8)
            self.encPoolSkip2 = nn.AvgPool3d(kernel_size=16, stride=16)

            self.encConv1 = nn.Sequential(
                nn.Conv3d(inChannel, eW, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )

            self.encConv2 = nn.Sequential(
                nn.Conv3d(eW + inChannel, 3*eW, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )

            self.encConv3 = nn.Sequential(
                nn.Conv3d(3*eW + inChannel, 6*eW, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.encConv4 = nn.Sequential(
                nn.Conv3d(6*eW + inChannel, 4*eW, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.encConv5 = nn.Sequential(
                nn.Conv3d(4*eW + inChannel, eW, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.encConv6 = nn.Sequential(
                nn.Conv3d(eW + inChannel, lat, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )

            self.encPool = nn.AdaptiveAvgPool3d(1)


    # encoder input shape: B S C W H (D) -> output shape: B S L
    def forward(self, data:torch.Tensor) -> torch.Tensor:
        if self.dimension == 2:
            d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
            #print(d.shape)

            skip1 = self.encPoolSkip1(d)
            #print(skip1.shape)
            skip2 = self.encPoolSkip2(d)
            #print(skip2.shape)

            d = self.encConv1(d)
            #print(d.shape)
            d = self.encConv2(torch.concat([d, skip1], dim=1))
            #print(d.shape)
            d = self.encConv3(torch.concat([d, skip2], dim=1))
            #print(d.shape)
            d = self.encConv4(torch.concat([d, skip2], dim=1))
            #print(d.shape)
            d = self.encConv5(torch.concat([d, skip2], dim=1))
            #print(d.shape)
            d = self.encConv6(torch.concat([d, skip2], dim=1))
            #print(d.shape)

            d = self.encPool(d)
            #print(d.shape)
            d = torch.squeeze(torch.squeeze(d, dim=3), dim=2)
            #print(d.shape)
            d = torch.reshape(d, (data.shape[0], data.shape[1], d.shape[1]))
            #print(d.shape)

        elif self.dimension == 3:
            d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4], data.shape[5]))
            #print(d.shape)

            skip1 = self.encPoolSkip1(d)
            #print(skip1.shape)
            skip2 = self.encPoolSkip2(d)
            #print(skip2.shape)

            d = self.encConv1(d)
            #print(d.shape)
            d = self.encConv2(torch.concat([d, skip1], dim=1))
            #print(d.shape)
            d = self.encConv3(torch.concat([d, skip2], dim=1))
            #print(d.shape)
            d = self.encConv4(torch.concat([d, skip2], dim=1))
            #print(d.shape)
            d = self.encConv5(torch.concat([d, skip2], dim=1))
            #print(d.shape)
            d = self.encConv6(torch.concat([d, skip2], dim=1))
            #print(d.shape)

            d = self.encPool(d)
            #print(d.shape)
            d = torch.squeeze(torch.squeeze(torch.squeeze(d, dim=4), dim=3), dim=2)
            #print(d.shape)
            d = torch.reshape(d, (data.shape[0], data.shape[1], d.shape[1]))
            #print(d.shape)
        return d




class DecoderModelSkip(nn.Module):
    p_me: ModelParamsEncoder
    p_md: ModelParamsDecoder
    p_ml: ModelParamsLatent

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder, p_md: ModelParamsDecoder, p_ml:ModelParamsLatent, dimension:int):
        super(DecoderModelSkip, self).__init__()

        self.p_d = p_d
        self.p_me = p_me
        self.p_md = p_md
        self.p_ml = p_ml
        self.dimension = dimension

        dW = self.p_md.decWidth
        sf = len(self.p_d.simFields)
        sp = len(self.p_d.simParams)
        lat = self.p_me.latentSize + sp
        if self.p_md.vae:
            lat = int(0.5 * lat)

        if self.dimension == 2:
            sizeX, sizeY = self.p_d.dataSize[0], self.p_d.dataSize[1]
            self.decConv1 = nn.Sequential(
                nn.Conv2d(lat, dW, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(4,2), mode='nearest'),
            )
            self.decConv2 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2,2), mode='nearest'),
            )
            self.decConv3 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2,2), mode='nearest'),
            )
            self.decConv4 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2,2), mode='nearest'),
            )
            self.decConv5 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(max(int(sizeX/2),64), max(int(sizeY/2),32)), mode='nearest'),
            )
            self.decConv6 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizeX, sizeY), mode='bilinear', align_corners=True),
            )
            self.decConv7 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
            )
            self.decConv8 = nn.Sequential(
                nn.Conv2d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dW, dimension + sf + sp, kernel_size=3, stride=1, padding=1),
            )

        elif self.dimension == 3:
            sizeX, sizeY, sizeZ = self.p_d.dataSize[0], self.p_d.dataSize[1], self.p_d.dataSize[2]
            self.decConv1 = nn.Sequential(
                nn.Conv3d(lat, dW, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(4,2,2), mode='nearest'),
            )
            self.decConv2 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2,2,2), mode='nearest'),
            )
            self.decConv3 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2,2,2), mode='nearest'),
            )
            self.decConv4 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2,2,2), mode='nearest'),
            )
            self.decConv5 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(max(int(sizeX/2),64), max(int(sizeY/2),32), max(int(sizeZ/2),32)), mode='nearest'),
            )
            self.decConv6 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizeX, sizeY, sizeZ), mode='trilinear', align_corners=True),
            )
            self.decConv7 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
            )
            self.decConv8 = nn.Sequential(
                nn.Conv3d(dW + lat, dW, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(dW, dimension + sf + sp, kernel_size=3, stride=1, padding=1),
            )


    # decoder input shape: B S L -> output shape: B S C W H (D)
    def forward(self, data:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        #print()
        #print(data.shape)
        #print(simParam.shape)
        seqLen = data.shape[1]

        # add simulation parameters to latent space
        d = torch.concat([data, simParam], dim=2) if simParam is not None else data
        #print(d.shape)
        if self.p_md.vae:
            d, vaeMean, vaeLogVar = reparameterizeAndSampleLatentSpaceVAE(d, simParam)
        else:
            vaeMean, vaeLogVar = None, None

        # combine batch and sequence dimension for decoder processing
        d = torch.reshape(d, (-1, d.shape[2]))

        if self.dimension == 2:
            lat = torch.unsqueeze(torch.unsqueeze(d, dim=2), dim=3)
            #print(lat.shape)

            d = self.decConv1(lat)
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv2(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv3(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv4(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv5(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv6(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv7(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3])
            d = self.decConv8(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            d = torch.reshape(d, (-1, seqLen, d.shape[1], d.shape[2], d.shape[3]))
            #print(d.shape)

        elif self.dimension == 3:
            lat = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(d, dim=2), dim=3), dim=4)
            #print(lat.shape)

            d = self.decConv1(lat)
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv2(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv3(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv4(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv5(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv6(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv7(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            latAdj = lat.expand(-1, -1, d.shape[2], d.shape[3], d.shape[4])
            d = self.decConv8(torch.concat([d, latAdj], dim=1))
            #print(d.shape)

            d = torch.reshape(d, (-1, seqLen, d.shape[1], d.shape[2], d.shape[3], d.shape[4]))
            #print(d.shape)
        return d, (vaeMean, vaeLogVar)



