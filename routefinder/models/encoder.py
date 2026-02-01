from typing import Callable, Optional

import torch.nn as nn

from torch import Tensor
import torchvision.models as models

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Sequential):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            feedforward_hidden: int = 512,
            normalization: Optional[str] = "batch",
            sdpa_fn: Optional[Callable] = None,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(MultiHeadAttention(embed_dim, num_heads, sdpa_fn=sdpa_fn)),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feedforward_hidden),
                    nn.ReLU(),
                    nn.Linear(feedforward_hidden, embed_dim),
                )
                if feedforward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class ImgGraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization, feed-forward layer, and normalization. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            num_layers: int,
            normalization: str = "batch",
            feedforward_hidden: int = 512,
            sdpa_fn: Optional[Callable] = None,
    ):
        super(ImgGraphAttentionNetwork, self).__init__()

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    embed_dim,
                    num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                )
                for _ in range(num_layers)
            )
        )
        self.resnet = BaseModel('resnet18')

    def forward(self, x: TensorDict, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x["coor"])
        h_img1 = self.resnet(x["img1"])
        h_img2 = self.resnet(x["img2"])
        return h


class BaseModel(nn.Module):
    def __init__(self, basename='resnet18', *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {}
        if basename == 'resnet18':
            self.basemodel = models.resnet18(pretrained=True)
            self.basemodel.layer2[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            self.basemodel.avgpool.register_forward_hook(self.get_activation('final_feature'))
        if basename == 'resnet50':
            self.basemodel = models.resnet50(pretrained=True)
            self.basemodel.layer2[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            self.basemodel.avgpool.register_forward_hook(self.get_activation('final_feature'))

    def get_activation(self, layer_name):
        def hook(module, input, output):
            self.output_feature[layer_name] = output

        return hook

    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['high_level_feature']
