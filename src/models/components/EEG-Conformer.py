import os

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
import warnings


class PatchEmbedding(nn.Module):
    """Patch Embedding.

    The authors used a convolution module to capture local features,
    instead of position embedding.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    n_channels: int
        Number of channels to be used as number of spatial filters.
    pool_time_length: int
        Length of temporal poling filter.
    stride_avg_pool: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.

    Returns
    -------
    x: torch.Tensor
        The output tensor of the patch embedding layer.
    """

    def __init__(
            self,
            n_filters_time,
            filter_time_length,
            n_channels,
            pool_time_length,
            stride_avg_pool,
            drop_prob,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time,
                      (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time,
                      (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, stride_avg_pool)
            ),
            # pooling acts as slicing to obtain 'patch' along the
            # time dimension as in ViT
            nn.Dropout(p=drop_prob),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class FullyConnected(nn.Module):
    def __init__(self, final_fc_length,
                 drop_prob_1=0.5, drop_prob_2=0.3, out_channels=256,
                 hidden_channels=32):
        """Fully-connected layer for the transformer encoder.

        Parameters
        ----------
        final_fc_length : int
            Length of the final fully connected layer.
        n_classes : int
            Number of classes for classification.
        drop_prob_1 : float
            Dropout probability for the first dropout layer.
        drop_prob_2 : float
            Dropout probability for the second dropout layer.
        out_channels : int
            Number of output channels for the first linear layer.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        add_log_softmax: bool
            Whether to add LogSoftmax non-linearity as the final layer.
        """

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class FinalLayer(nn.Module):
    def __init__(self, n_classes, hidden_channels=32, return_features=False, add_log_softmax=True):
        """Classification head for the transformer encoder.

        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        add_log_softmax : bool
            Adding LogSoftmax or not.
        """

        super().__init__()
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_channels, n_classes),
        )
        self.return_features = return_features
        if add_log_softmax:
            classification = nn.LogSoftmax(dim=1)
        else:
            classification = nn.Identity()
        if not self.return_features:
            self.final_layer.add_module("classification", classification)

    def forward(self, x):
        if self.return_features:
            out = self.final_layer(x)
            return out, x
        else:
            out = self.final_layer(x)
            return out


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, att_heads, att_drop, forward_expansion=4):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion,
                        drop_p=att_drop
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.

    Similar to the layers used in ViT.

    Parameters
    ----------
    att_depth : int
        Number of transformer encoder blocks.
    emb_size : int
        Embedding size of the transformer encoder.
    att_heads : int
        Number of attention heads.
    att_drop : float
        Dropout probability for the attention layers.

    """

    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(
            *[
                TransformerEncoderBlock(emb_size, att_heads, att_drop)
                for _ in range(att_depth)
            ]
        )


class EEGConformer(nn.Module):
    """EEG Conformer.

    Convolutional Transformer for EEG decoding.

    The paper and original code with more details about the methodological
    choices are available at the [Song2022]_ and [ConformerCode]_.

    This neural network architecture receives a traditional braindecode input.
    The input shape should be three-dimensional matrix representing the EEG
    signals.

         `(batch_size, n_channels, n_timesteps)`.

    The EEG Conformer architecture is composed of three modules:
        - PatchEmbedding
        - TransformerEncoder
        - ClassificationHead

    Notes
    -----
    The authors recommend using data augmentation before using Conformer,
    e.g. segmentation and recombination,
    Please refer to the original paper and code for more details.

    The model was initially tuned on 4 seconds of 250 Hz data.
    Please adjust the scale of the temporal convolutional layer,
    and the pooling layer for better performance.

    .. versionadded:: 0.8

    We aggregate the parameters based on the parts of the models, or
    when the parameters were used first, e.g. n_filters_time.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.
    att_depth: int
        Number of self-attention layers.
    att_heads: int
        Number of attention heads.
    att_drop_prob: float
        Dropout rate of the self-attention layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.
    return_features: bool
        If True, the forward method returns the features before the
        last classification layer. Defaults to False.
    n_classes :
        Alias for n_outputs.
    n_channels :
        Alias for n_chans.
    input_window_samples :
        Alias for n_times.
    References
    ----------
    .. [Song2022] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, pp.710-719. https://ieeexplore.ieee.org/document/9991178
    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """

    def __init__(
            self,
            n_outputs=None,
            n_chans=None,
            n_filters_time=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            drop_prob=0.5,
            att_depth=6,
            att_heads=10,
            att_drop_prob=0.5,
            final_fc_length=2440,
            return_features=False,
            n_times=None,
            add_log_softmax=True, ):

        self.n_outputs = n_outputs
        self.n_chans = n_chans
        self.n_times = n_times
        self.add_log_softmax = add_log_softmax

        super().__init__()

        if not (self.n_chans <= 64):
            warnings.warn("This model has only been tested on no more " +
                          "than 64 channels. no guarantee to work with " +
                          "more channels.", UserWarning)

        self.patch_embedding = PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob)

        if final_fc_length == "auto":
            assert self.n_times is not None
            final_fc_length = self.get_fc_size()

        self.transformer = TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob)

        self.fc = FullyConnected(
            final_fc_length=final_fc_length)

        self.final_layer = FinalLayer(n_classes=self.n_outputs,
                                      return_features=return_features,
                                      add_log_softmax=self.add_log_softmax)

    def forward(self, x: Tensor) -> Tensor:
        # x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = self.final_layer(x)
        return x

    def get_fc_size(self):

        out = self.patch_embedding(torch.ones((1, 1,
                                               self.n_chans,
                                               self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]

        return size_embedding_1 * size_embedding_2


if __name__ == '__main__':
    gpus = [0]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
    import math

    # model = Conformer(emb_kernel_size_1=[1, 25], emb_kernel_size_2=[22, 1]).cuda()

    # model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
    # model = model.cuda()

    # # print(model)
    # # x = torch.randn(200, 1, 22, 1000).cuda()
    #
    # tensor_cpu = torch.rand(5, 5)  # This creates a 5x5 tensor with random values between 0 and 1.
    #
    # # Move the tensor to the GPU and cast to torch.cuda.FloatTensor
    # tensor_cuda = tensor_cpu.type(torch.cuda.FloatTensor)
    #
    # summary(model, (1, 22, 1000))
    #
    # out = model(x)
    # model_name = 'eegconformer-d1.onnx'
    # model_path_onnx = os.path.join("C:\\Users\\alioo\\Desktop\\MA\\bbcpy_AutoML\\local", model_name)
    # torch.onnx.export(model, x, model_path_onnx, input_names=["features"], output_names=["logits"])
    # print(out.shape)
