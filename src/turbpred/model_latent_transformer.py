import math
import torch
import torch.nn as nn

from turbpred.params import DataParams, ModelParamsEncoder, ModelParamsLatent



class PositionalEncoding(nn.Module):
    def __init__(self, latSize:int, dropout:float=0.0, maxLen:int=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(maxLen).unsqueeze(1)
        div = torch.exp(torch.arange(0, latSize, 2) * (-math.log(10000.0) / latSize)).unsqueeze(0)
        pe = torch.zeros(1, maxLen, latSize)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    # input shape: B S L
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #print("\tPOS ENCODING:")
        #print("\t" + str(x.shape))
        x = x + self.pe[:, :x.shape[1]]
        #print("\t" + str(x.shape))
        return self.dropout(x)




class LatentModelTransformerEnc(nn.Module):
    p_d: DataParams
    p_me: ModelParamsEncoder
    p_ml: ModelParamsLatent

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder, p_ml:ModelParamsLatent, flatLatent:bool):
        super(LatentModelTransformerEnc, self).__init__()

        self.p_d = p_d
        self.p_me = p_me
        self.p_ml = p_ml
        self.flatLatent = flatLatent

        lat = self.p_me.latentSize + len(self.p_d.simParams) if not flatLatent else 1
        width = self.p_ml.width
        layers = self.p_ml.layers
        heads = self.p_ml.heads if not flatLatent else 1
        dropout = self.p_ml.dropout

        self.posEncoding = PositionalEncoding(lat, dropout)
        encoderLayer = nn.TransformerEncoderLayer(lat, heads, width, dropout, batch_first=True)
        self.transformerEnc = nn.TransformerEncoder(encoderLayer, layers)

    # input shape: B S L -> output shape: B S L
    def forward(self, data:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        #print()
        #print(data.shape)
        #print(simParam.shape)

        # add simulation parameters number to latent space
        seqLen = data.shape[1]
        d = torch.concat([data, simParam], dim=2) if simParam is not None else data
        if self.flatLatent:
            seqLen = data.shape[1] * (data.shape[2]+1)
            d = torch.reshape(d, (d.shape[0], seqLen, 1))

        # prepare mask
        mask = torch.triu(torch.ones(seqLen, seqLen) * float('-inf'), diagonal=1)
        mask = mask.to("cuda" if data.is_cuda else "cpu")

        #print(d.shape, mask.shape)
        d = self.posEncoding(d)
        #print(d.shape)
        d = self.transformerEnc(d, mask)
        #print(d.shape)

        if self.flatLatent:
            d = torch.reshape(d, (data.shape[0], data.shape[1], data.shape[2]+1))

        if self.p_d.simParams:
            return d[:, :, :-len(self.p_d.simParams)] #remove simulation parameter prediction
        else:
            return d



class LatentModelTransformerDec(nn.Module):
    p_d: DataParams
    p_me: ModelParamsEncoder
    p_ml: ModelParamsLatent

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder, p_ml:ModelParamsLatent):
        super(LatentModelTransformerDec, self).__init__()

        self.p_d = p_d
        self.p_me = p_me
        self.p_ml = p_ml

        lat = self.p_me.latentSize + len(self.p_d.simParams)
        width = self.p_ml.width
        layers = self.p_ml.layers
        heads = self.p_ml.heads
        dropout = self.p_ml.dropout

        self.posEncoding = PositionalEncoding(lat, dropout)
        decoderLayer = nn.TransformerDecoderLayer(lat, heads, width, dropout, batch_first=True)
        self.transformerDec = nn.TransformerDecoder(decoderLayer, layers)

    # input shapes: B S L -> output shape: B S L
    def forward(self, data:torch.Tensor, target:torch.Tensor, simParamData:torch.Tensor, simParamTarget:torch.Tensor) -> torch.Tensor:
        #print()
        #print(data.shape)
        #print(target.shape)
        #print(simParamData.shape)
        #print(simParamTarget.shape)

        # add simulation parameters to latent spaces
        seqLenData, seqLenTarget = data.shape[1], target.shape[1]
        d = torch.concat([data, simParamData], dim=2) if simParamData is not None else data
        t = torch.concat([target, simParamTarget], dim=2) if simParamTarget is not None else target

        # prepare masks
        maskData   = torch.triu(torch.ones(seqLenTarget, seqLenData) * float('-inf'), diagonal=1) # note: correct mask shape!
        maskData   = maskData.to("cuda" if data.is_cuda else "cpu")
        maskTarget = torch.triu(torch.ones(seqLenTarget, seqLenTarget) * float('-inf'), diagonal=1)
        maskTarget = maskTarget.to("cuda" if data.is_cuda else "cpu")

        #print(data.shape, maskData.shape, target.shape, maskTarget.shape)
        d = self.posEncoding(d)
        t = self.posEncoding(t)
        #print(d.shape)
        d = self.transformerDec(tgt=t, memory=d, tgt_mask=maskTarget, memory_mask=maskData)
        #print(d.shape)
        if self.p_d.simParams:
            return d[:, :, :-len(self.p_d.simParams)] #remove simulation parameter prediction
        else:
            return d



class LatentModelTransformer(nn.Module):
    p_d: DataParams
    p_me: ModelParamsEncoder
    p_ml: ModelParamsLatent

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder, p_ml:ModelParamsLatent):
        super(LatentModelTransformer, self).__init__()

        self.p_d = p_d
        self.p_me = p_me
        self.p_ml = p_ml

        lat = self.p_me.latentSize + len(self.p_d.simParams)
        width = self.p_ml.width
        layers = self.p_ml.layers
        heads = self.p_ml.heads
        dropout = self.p_ml.dropout

        self.posEncoding = PositionalEncoding(lat, dropout)
        self.transformer = nn.Transformer(lat, heads, int(layers/2), int(layers/2), width, dropout, batch_first=True)

    # input shapes: B S L -> output shape: B S L
    def forward(self, data:torch.Tensor, target:torch.Tensor, simParamData:torch.Tensor, simParamTarget:torch.Tensor) -> torch.Tensor:
        #print()
        #print(data.shape)
        #print(target.shape)
        #print(simParamData.shape)
        #print(simParamTarget.shape)

        # add simulation parameters to latent spaces
        seqLenData, seqLenTarget = data.shape[1], target.shape[1]
        d = torch.concat([data, simParamData], dim=2) if simParamData is not None else None
        t = torch.concat([target, simParamTarget], dim=2) if simParamTarget is not None else None

        # prepare masks
        maskData   = torch.triu(torch.ones(seqLenData, seqLenData) * float('-inf'), diagonal=1) # note: different mask shape!
        maskData   = maskData.to("cuda" if data.is_cuda else "cpu")
        maskTarget = torch.triu(torch.ones(seqLenTarget, seqLenTarget) * float('-inf'), diagonal=1)
        maskTarget = maskTarget.to("cuda" if data.is_cuda else "cpu")

        #print(d.shape, maskData.shape, t.shape, maskTarget.shape)
        d = self.posEncoding(d)
        t = self.posEncoding(t)
        #print(d.shape)
        d = self.transformer(src=d, tgt=t, src_mask=maskData, tgt_mask=maskTarget)
        #print(d.shape)
        if self.p_d.simParams:
            return d[:, :, :-len(self.p_d.simParams)] #remove simulation parameter prediction
        else:
            return d



class LatentModelTransformerMGN(nn.Module):
    p_d: DataParams
    p_me: ModelParamsEncoder
    p_ml: ModelParamsLatent

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder, p_ml:ModelParamsLatent):
        super(LatentModelTransformerMGN, self).__init__()

        self.p_d = p_d
        self.p_me = p_me
        self.p_ml = p_ml

        lat = self.p_me.latentSize
        width = self.p_ml.width
        layers = self.p_ml.layers
        heads = self.p_ml.heads
        dropout = self.p_ml.dropout

        self.posEncoding = PositionalEncoding(lat, dropout)
        decoderLayer = nn.TransformerDecoderLayer(lat, heads, width, dropout, batch_first=True)
        self.transformerDec = nn.TransformerDecoder(decoderLayer, layers)

    # input shapes: B S L -> output shape: B S L
    def forward(self, data:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        #print()
        #print(data.shape)
        #print(target.shape)

        # add simulation parameters to latent spaces
        seqLenData, seqLenTarget = data.shape[1], target.shape[1]

        # prepare masks
        maskData   = torch.triu(torch.ones(seqLenTarget, seqLenData) * float('-inf'), diagonal=1) # note: correct mask shape!
        maskData   = maskData.to("cuda" if data.is_cuda else "cpu")
        maskTarget = torch.triu(torch.ones(seqLenTarget, seqLenTarget) * float('-inf'), diagonal=1)
        maskTarget = maskTarget.to("cuda" if data.is_cuda else "cpu")

        #print(data.shape, maskData.shape, target.shape, maskTarget.shape)
        d = self.posEncoding(data)
        t = self.posEncoding(target)
        #print(d.shape)
        d = self.transformerDec(tgt=t, memory=d, tgt_mask=maskTarget, memory_mask=maskData)
        #print(d.shape)
        return d #remove simulation parameter prediction


class LatentModelTransformerMGNParamEmb(nn.Module):
    p_d: DataParams
    p_me: ModelParamsEncoder

    def __init__(self, p_d:DataParams, p_me:ModelParamsEncoder):
        super(LatentModelTransformerMGNParamEmb, self).__init__()

        self.p_d = p_d
        self.p_me = p_me

        lat = self.p_me.latentSize

        self.paramEmb = nn.Sequential(
            nn.Linear(len(self.p_d.simParams), 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, lat),
        )

    def forward(self, simParam:torch.Tensor) -> torch.Tensor:
        #print()
        #print(simParam.shape)

        return self.paramEmb(simParam)

