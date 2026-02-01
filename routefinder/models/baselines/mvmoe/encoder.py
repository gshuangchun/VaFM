import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)
from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork
from rl4co.models.nn.ops import Normalization
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor
from einops import rearrange
from routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbedding

from .moe import MoE
import torchvision.models as models
from tensordict.tensordict import TensorDict
from typing import Callable, Optional
import torch.nn.functional as F

log = get_pylogger(__name__)


def scaled_dot_product_attention_simple(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
        "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
    )
    scaled_dot_product_attention = scaled_dot_product_attention_simple


class MVMoEInitEmbedding(MTVRPInitEmbedding):
    def __init__(
            self,
            embed_dim=128,
            num_experts=4,
            routing_method="input_choice",
            routing_level="node",
            topk=2,
            bias=False,
            **kw,
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MVMoEInitEmbedding, self).__init__(embed_dim, bias, **kw)

        # If MoE is provided, we re-initialize the projections with MoE
        if num_experts > 0:
            print("MoE in init embedding initializing")
            self.project_global_feats = MoE(
                input_size=2,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="Linear",
            )
            self.project_customers_feats = MoE(
                input_size=7,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="Linear",
            )

    def forward(self, td):
        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        global_feats = td["locs"][:, :1, :]

        # Customers (batch, N, 5) -> (batch, N, embed_dim)
        # note that these feats include the depot (but unused) so we exclude the first node
        cust_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ),
            -1,
        )

        # If some features are infinity (e.g. distance limit is inf because of no limit), replace with 0 so that it does not affect the embedding
        global_feats = torch.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.nan_to_num(cust_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # MoE loss is 0 if layer is not MoE
        moe_loss_global, moe_loss_cust = 0, 0
        if isinstance(self.project_global_feats, MoE):
            global_embeds, moe_loss_global = self.project_global_feats(global_feats)
        else:
            global_embeds = self.project_global_feats(global_feats)
        if isinstance(self.project_customers_feats, MoE):
            cust_embeds, moe_loss_cust = self.project_customers_feats(cust_feats)
        else:
            cust_embeds = self.project_customers_feats(cust_feats)
        self.moe_loss = moe_loss_global + moe_loss_cust
        return torch.cat((global_embeds, cust_embeds), -2)


class MultiHeadAttentionLayerMoE(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            feedforward_hidden: int = 512,
            normalization="instance",
            sdpa_fn=None,
            num_experts=4,
            routing_method="input_choice",
            routing_level="node",
            topk=2,
    ):
        super(MultiHeadAttentionLayerMoE, self).__init__()

        if num_experts > 0:
            print("MoE in MultiHeadAttentionLayer initializing")
            dense_net = MoE(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=num_experts,
                hidden_size=feedforward_hidden,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                moe_model="MLP",
            )
        else:
            dense_net = nn.Sequential(
                nn.Linear(embed_dim, feedforward_hidden),
                nn.ReLU(),
                nn.Linear(feedforward_hidden, embed_dim),
            )

        self.mha = MultiHeadAttention(embed_dim, num_heads, sdpa_fn=sdpa_fn)
        self.norm1 = Normalization(embed_dim, normalization)
        self.dense = dense_net
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor) -> Tensor:
        out_mha = self.mha(x)
        h = out_mha + x  # skip connection
        h = self.norm1(h)
        moe_loss = 0
        if isinstance(self.dense, MoE):
            out_dense, moe_loss = self.dense(h)
        else:
            out_dense = self.dense(h)
        # save moe loss
        self.moe_loss = moe_loss
        h = out_dense + h  # skip connection
        h = self.norm2(h)
        return h


class GraphAttentionNetworkMVMoE(GraphAttentionNetwork):
    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            num_layers: int,
            normalization: str = "instance",
            feedforward_hidden: int = 512,
            sdpa_fn=None,
            moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
            num_experts=4,
            routing_method="input_choice",
            routing_level="node",
            topk=2,
    ):
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayerMoE(
                    embed_dim,
                    num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                    num_experts=0 if f"enc{i}" not in moe_loc else num_experts,
                    routing_method=routing_method,
                    routing_level=routing_level,
                    topk=topk,
                )
                for i in range(num_layers)
            )
        )
        self.K = 65536
        self.dim = 128
        self.dim_vis = 512
        self.dim_vis_l = 256
        self.resnet = BaseModel('resnet18')
        self.num_att = 4
        self.conv_h = nn.Conv2d(self.dim_vis, self.dim, kernel_size=1)
        self.conv_l = nn.Conv2d(self.dim_vis_l, self.dim, kernel_size=1)
        self.conv_f = nn.Conv2d(self.dim_vis, self.dim, kernel_size=1)
        # # self.mlp_img_att = nn.Linear((512), 128)
        # self.mlp_img_final = nn.Linear((128 + 128), 128)
        # self.mlp_img1 = nn.Linear((512), 128)
        self.mlp_img2 = nn.Linear(self.dim, self.num_att)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.loss_mse_ft = nn.MSELoss()
        self.loss_cls_ft = nn.BCEWithLogitsLoss()
        self.loss_proto_ft = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        # self.mha = MultiHeadAttentionVision(embed_dim, num_heads, sdpa_fn=sdpa_fn)
        self.mha_spa = MultiHeadAttentionVisionSpatial(embed_dim, num_heads, sdpa_fn=sdpa_fn)
        self.mha_spa_l = MultiHeadAttentionVisionSpatial(embed_dim, num_heads, sdpa_fn=sdpa_fn)

        # # create the queue
        # self.register_buffer("queue", torch.randn(self.dim, self.K))
        #
        # self.register_buffer("O_prototpye", torch.rand(2, self.dim))
        # self.register_buffer("L_prototpye", torch.rand(2, self.dim))
        # self.register_buffer("B_prototpye", torch.rand(2, self.dim))
        # self.register_buffer("TW_prototpye", torch.rand(2, self.dim))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.register_buffer("queue_l_O", torch.zeros(1, self.K).long())
        # self.register_buffer("queue_l_L", torch.zeros(1, self.K).long())
        # self.register_buffer("queue_l_B", torch.zeros(1, self.K).long())
        # self.register_buffer("queue_l_TW", torch.zeros(1, self.K).long())
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels_B, labels_O, labels_L, labels_TW):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size, :]
            labels_O = labels_O[:batch_size, :]
            labels_L = labels_L[:batch_size, :]
            labels_B = labels_B[:batch_size, :]
            labels_TW = labels_TW[:batch_size, :]

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_l_O[:, ptr:ptr + batch_size] = labels_O.T
        self.queue_l_L[:, ptr:ptr + batch_size] = labels_L.T
        self.queue_l_B[:, ptr:ptr + batch_size] = labels_B.T
        self.queue_l_TW[:, ptr:ptr + batch_size] = labels_TW.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x: TensorDict, mask=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x["cord"])
        # x_img = x["img1"]
        x_img = torch.cat([x["img1"], x["img2"]], dim=1)
        h_img1, l_img1 = self.resnet(x_img)
        global_feats = self.avgpool(self.conv_f(h_img1)).squeeze(-1).squeeze(-1)
        logit_cls = self.mlp_img2(global_feats)
        all_cls_label = x["label"][:, :, 1:].sum(-2)
        all_cls_label[all_cls_label > 0] = 1
        self.loss_cls = self.loss_cls_ft(logit_cls, all_cls_label)

        B, S, D = h.shape
        h_node = self.conv_h(h_img1)
        node_img_feats = self.get_node_feats(h_node, x["final_block_indices"], h)
        l_node = self.conv_l(l_img1)
        node_img_feats_l1 = self.get_node_feats(l_node, x["final_block_indices_l"], h)
        # self.idx_NTW = torch.where(all_cls_label[:, -1] == 0)[0]
        # if len(self.idx_NTW) > 0:
        #     node_img_feats[self.idx_NTW] = node_img_feats_l1[self.idx_NTW]

        # node_img_feats_l1 = self.get_node_feats(self.conv_l(l_img1), x["final_block_indices_l"], h)
        # encoded_nodes_img = self.mlp_img(node_img_feats_h1)
        # encoded_nodes_img = node_img_feats_h1 + node_img_feats_l1
        # self.loss_mse = self.loss_mse_ft(h.reshape(-1, D), encoded_nodes_img)
        h_final_h = self.mha_spa(h + node_img_feats, h_node)
        h_final_l = self.mha_spa_l(h + node_img_feats_l1, h_node)
        h_final = torch.stack([h_final_h, h_final_l], dim=0)

        # self.O_prototpye = torch.vstack(
        #     [self.queue[:, torch.where(self.queue_l_O == i)[1]].mean(axis=1) if len(
        #         torch.where(self.queue_l_O == i)[1]) > 0 else self.O_prototpye[i, :] for i in
        #      range(2)])
        # self.L_prototpye = torch.vstack(
        #     [self.queue[:, torch.where(self.queue_l_L == i)[1]].mean(axis=1) if len(
        #         torch.where(self.queue_l_L == i)[1]) > 0 else self.L_prototpye[i, :] for i in
        #      range(2)])
        # self.B_prototpye = torch.vstack(
        #     [self.queue[:, torch.where(self.queue_l_B == i)[1]].mean(axis=1) if len(
        #         torch.where(self.queue_l_B == i)[1]) > 0 else self.B_prototpye[i, :] for i in
        #      range(2)])
        # self.TW_prototpye = torch.vstack(
        #     [self.queue[:, torch.where(self.queue_l_TW == i)[1]].mean(axis=1) if len(
        #         torch.where(self.queue_l_TW == i)[1]) > 0 else self.TW_prototpye[i, :] for i in
        #      range(2)])
        #
        # l_proto_O = torch.einsum('nc,cm->nm', [h_final[:, 1:, :].reshape(-1, D), self.O_prototpye.T])
        # l_proto_L = torch.einsum('nc,cm->nm', [h_final[:, 1:, :].reshape(-1, D), self.L_prototpye.T])
        # l_proto_B = torch.einsum('nc,cm->nm', [h_final[:, 1:, :].reshape(-1, D), self.B_prototpye.T])
        # l_proto_TW = torch.einsum('nc,cm->nm', [h_final[:, 1:, :].reshape(-1, D), self.TW_prototpye.T])
        #
        # label_nodes = x["label"][:, :, 1:].view(-1, self.num_att).long()
        # loss_B = self.loss_proto_ft(l_proto_B, nn.functional.one_hot(label_nodes[:, 0], num_classes=2).float())
        # loss_O = self.loss_proto_ft(l_proto_O, nn.functional.one_hot(label_nodes[:, 1], num_classes=2).float())
        # loss_L = self.loss_proto_ft(l_proto_L, nn.functional.one_hot(label_nodes[:, 2], num_classes=2).float())
        # loss_TW = self.loss_proto_ft(l_proto_TW, nn.functional.one_hot(label_nodes[:, 3], num_classes=2).float())
        # self.loss_proto = (loss_B + loss_O + loss_L + loss_TW) / 4
        # self._dequeue_and_enqueue(global_feats, all_cls_label[:, 0:1], all_cls_label[:, 1:2], all_cls_label[:, 2:3],
        #                           all_cls_label[:, 3:4])

        # high_feats = self.avgpool(h_img1).squeeze(-1).squeeze(-1)
        # logit_cls = self.mlp_img1(high_feats)
        # cls_label = x_img.view(B, 6, -1).max(dim=-1).values
        # cls_label = torch.where(cls_label > 0, torch.tensor(1, device=cls_label.device, dtype=cls_label.dtype),
        #                         cls_label)
        # cls_tw_label = cls_label[:, 3:].max(dim=-1).values
        # all_cls_label = torch.cat((cls_label[:, :3], cls_tw_label[:, None]), dim=-1)
        # self.loss_cls = self.loss_cls_ft(logit_cls, all_cls_label)
        #
        # para_weights = self.mlp_img1.weight.clone().transpose(0, 1)
        # img_feat_h = self.get_att_feats(h_img1, all_cls_label, logit_cls, para_weights[:h_img1.shape[1], :])
        # B, S, D = h.shape
        # node_img_feats_h1_att = self.get_node_feats(img_feat_h, x["final_block_indices"], h)
        # encoded_nodes_img_att = self.mlp_img(node_img_feats_h1_att).view(B, S, -1)
        # encoded_nodes_img_att = encoded_nodes_img.clone().view(B, S, -1)

        return h_final

    def get_att_feats(self, img1, all_cls_label, logit_h, para_weights):
        att_weights = self.sigmoid(para_weights)
        B, D, H, W = img1.shape
        h_img1_att = img1[:, :, None, :, :].expand(B, D, self.num_att, H, W) * att_weights[None, :, :, None, None]
        need_prob = all_cls_label - self.sigmoid(logit_h)
        # need_prob[need_prob < 0] = float('-inf')
        f_prob = torch.softmax(torch.where(need_prob < 0, torch.tensor(float('-inf')), need_prob), dim=-1)
        img_feat_h = h_img1_att * f_prob[:, None, :, None, None]
        img_feat_h = img_feat_h.sum(dim=2)
        return img_feat_h

    def get_node_feats(self, encoded_nodes_img, indices, h):
        B, S, _ = h.shape
        _, D, H, W = encoded_nodes_img.shape
        flatten_img_feats = encoded_nodes_img.view(B, D, -1)[:, None, :, :].expand(B, S, D, H * W)
        node_img_feats = flatten_img_feats.gather(dim=-1, index=indices[:, :, None, None].expand(B, S, D, H * W).to(
            torch.int64))[:, :, :, 0]
        return node_img_feats


class MVMoEEncoder(AttentionModelEncoder):
    def __init__(
            self,
            embed_dim: int = 128,
            num_heads: int = 8,
            num_layers: int = 6,
            normalization: str = "instance",
            feedforward_hidden: int = 512,
            env_name="mtvrp",
            sdpa_fn=None,
            init_embedding=None,
            num_experts=4,
            routing_method="input_choice",
            routing_level="node",
            topk=2,
            moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
            **unused,
    ):
        # super(MVMoEEncoder, self).__init__()
        nn.Module.__init__(self)

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        assert self.env_name == "mtvrp", "Only mtvrp is supported for MVMoE"

        # assert init_embedding is None, "init embedding is manually set in MVMoE"

        # Initialize raw features only if provided
        if "raw" in moe_loc:
            num_experts_init = num_experts
        else:
            num_experts_init = 0

        if not init_embedding:
            init_embedding = MVMoEInitEmbedding(
                embed_dim,
                num_experts=num_experts_init,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
            )
        else:
            if num_experts_init > 0:
                log.warning("MoE requested for init embedding but already provided")
        self.init_embedding = init_embedding

        self.net = GraphAttentionNetworkMVMoE(
            num_heads,
            embed_dim,
            num_layers,
            normalization,
            feedforward_hidden,
            sdpa_fn=sdpa_fn,
            moe_loc=moe_loc,
            num_experts=num_experts,
            routing_method=routing_method,
            routing_level=routing_level,
            topk=topk,
        )


class BaseModel(nn.Module):
    def __init__(self, basename='resnet18', *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {}
        if basename == 'resnet18':
            self.basemodel = models.resnet18(pretrained=True)
            self.basemodel.layer3[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            self.basemodel.avgpool.register_forward_hook(self.get_activation('final_feature'))
        if basename == 'resnet50':
            self.basemodel = models.resnet50(pretrained=True)
            self.basemodel.layer3[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            self.basemodel.avgpool.register_forward_hook(self.get_activation('final_feature'))
        original_conv1 = self.basemodel.conv1

        self.basemodel.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        # 初始化新的通道的权重
        with torch.no_grad():
            # 将新通道的权重初始化为原始权重的均值
            self.basemodel.conv1.weight[:, :3] = original_conv1.weight
            self.basemodel.conv1.weight[:, 3:] = original_conv1.weight.mean(dim=1, keepdim=True)

    def hook(self, module, input, output, layer_name):
        self.output_feature[layer_name] = output

    def get_activation(self, layer_name):
        # 使用functools.partial绑定参数layer_name
        from functools import partial
        return partial(self.hook, layer_name=layer_name)

    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['high_level_feature'], self.output_feature['low_level_feature']


class MultiHeadAttentionVision(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            attention_dropout: float = 0.0,
            causal: bool = False,
            device: str = None,
            dtype: torch.dtype = None,
            sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention_simple

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wkv = nn.Linear(512, 2 * 512, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, h, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        B, S, _ = x.shape

        f_vis = h
        f_coor = x.transpose(1, 2)
        q = rearrange(
            f_coor, "b s (one h d) -> one b h s d", one=1, h=1
        ).unbind(dim=0)[0]
        k, v = rearrange(
            self.Wkv(f_vis), "b s (two h d) -> two b h s d", two=2, h=1
        ).unbind(dim=0)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)").transpose(1, 2))


class MultiHeadAttentionVisionSpatial(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            attention_dropout: float = 0.0,
            causal: bool = False,
            device: str = None,
            dtype: torch.dtype = None,
            sdpa_fn: Optional[Callable] = None,
            normalization="instance",
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.feedforward_hidden = 512
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention_simple

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, self.feedforward_hidden),
            nn.ReLU(),
            nn.Linear(self.feedforward_hidden, embed_dim),
        )
        self.norm1 = Normalization(embed_dim, normalization)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x, h, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        B, S, D = x.shape

        f_vis = h
        f_coor = x
        q = rearrange(
            f_coor, "b s (one h d) -> one b h s d", one=1, h=self.num_heads
        ).unbind(dim=0)[0]
        k, v = rearrange(
            self.Wkv(f_vis.view(B, D, -1).transpose(1, 2)), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.attention_dropout,
        )
        h = x + self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
        h = self.norm1(h)
        out_dense = self.dense(h)
        h = out_dense + h  # skip connection
        h = self.norm2(h)
        return h
