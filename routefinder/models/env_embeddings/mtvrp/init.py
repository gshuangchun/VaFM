import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict


class MTVRPInitEmbedding(nn.Module):
    """Initial embedding MTVRP.

    Note: this is the same as what MTPOMO and MVMoE use.

    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - durations: duration of the nodes (TW)

    Global features:
        - loc: x, y euclidean coordinates of depot
    """

    def __init__(
            self, embed_dim=128, bias=False, **kw
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MTVRPInitEmbedding, self).__init__()

        # Depot feats (includes global features): x, y, distance, backhaul_class, open_route
        global_feat_dim = 2
        self.project_global_feats = nn.Linear(global_feat_dim, embed_dim, bias=bias)

        # Customer feats: x, y, demand_linehaul, demand_backhaul, time_window_early, time_window_late, durations
        customer_feat_dim = 7
        self.project_customers_feats = nn.Linear(customer_feat_dim, embed_dim, bias=bias)

        self.embed_dim = embed_dim

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
        global_embeddings = self.project_global_feats(
            global_feats
        )  # [batch, 1, embed_dim]
        cust_embeddings = self.project_customers_feats(
            cust_feats
        )  # [batch, N, embed_dim]
        return torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, N+1, embed_dim]


# Note that this is the most recent version and should be used from now on. Others can be based on this!
class MTVRPInitEmbeddingRouteFinderBase(nn.Module):
    """General Init embedding class

    Args:
        num_global_feats: number of global features
        num_cust_feats: number of customer features
        embed_dim: embedding dimension
        bias: whether to use bias in the linear layers
        posinf_val: value to replace positive infinity values with
    """

    def __init__(
            self, num_global_feats, num_cust_feats, embed_dim=128, bias=False, posinf_val=0.0
    ):
        super(MTVRPInitEmbeddingRouteFinderBase, self).__init__()
        self.project_global_feats = nn.Linear(num_global_feats, embed_dim, bias=bias)
        self.project_customers_feats = nn.Linear(num_cust_feats, embed_dim, bias=bias)
        self.embed_dim = embed_dim
        self.posinf_val = posinf_val

    def _global_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _cust_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def forward(self, td):
        global_feats = self._global_feats(td)
        cust_feats = self._cust_feats(td)
        imgs1, final_block_indices, final_block_indices_l = self.get_problems_img(td.shape[0],
                                                                                  cust_feats.shape[1], td)
        imgs2 = self.get_problems_img_LTW(td.shape[0], cust_feats.shape[1], td)

        global_feats = torch.nan_to_num(global_feats, posinf=self.posinf_val)
        cust_feats = torch.nan_to_num(cust_feats, posinf=self.posinf_val)
        label_BC = cust_feats[:, :, 2:4]
        label_BC[label_BC > 0] = 1
        label_TW = cust_feats[:, :, 4:7]
        label_TW = label_TW.sum(dim=-1)
        label_TW[label_TW > 0] = 1
        label_TW = label_TW.unsqueeze(dim=-1)
        label_OL = global_feats.expand(label_BC.shape[0], label_BC.shape[1], global_feats.shape[2])[:, :, 0:2]
        label_OL[label_OL > 0] = 1
        label_node = torch.cat((label_BC, label_OL, label_TW), -1)

        global_embeddings = self.project_global_feats(global_feats)
        cust_embeddings = self.project_customers_feats(cust_feats)
        h = TensorDict(
            {
                "cord": torch.cat((global_embeddings, cust_embeddings), -2),
                "img1": imgs1,
                "img2": imgs2,
                "final_block_indices": final_block_indices,
                "final_block_indices_l": final_block_indices_l,
                "label": label_node,
            }
        )

        return h

    def get_problems_img(self, batch_size, problem_size, td):
        problem_size = problem_size + 1
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        problems = td["locs"]
        patches = 7
        patches_l = 14
        img_size = 224
        patch_size = img_size // patches
        patch_size_l = img_size // patches_l

        if problem_size > 100:
            offsets = torch.tensor([[0, 0], [0, 1], [0, -1],
                                    [1, -1], [1, 0], [1, 1],
                                    [-1, -1], [-1, 0], [-1, 1]])
            offsets_m = torch.tensor([3, 5,
                                      6, 8])
        else:
            offsets = torch.tensor([[0, 0], [0, 1], [0, 2], [0, -2], [0, -1],
                                    [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                    [2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
                                    [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
                                    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2]])
            offsets_m = torch.tensor([5, 6, 8, 9,
                                      10, 11, 13, 14,
                                      15, 16, 18, 19,
                                      20, 21, 23, 24])
        bg = td["open_route"].float()
        bg[bg == 0] = 0.25
        bg[bg == 1] = 0.75
        imgs = torch.ones((batch_size, 3, img_size, img_size)) * bg.unsqueeze(-1).unsqueeze(-1)
        xy_img = problems * 200 + 12
        xy_img = xy_img.int()
        block_indices = xy_img // patch_size
        final_block_indices = block_indices[:, :, 0] * patches + block_indices[:, :, 1]
        block_indices_l = xy_img // patch_size_l
        final_block_indices_l = block_indices_l[:, :, 0] * patches + block_indices_l[:, :, 1]
        xy_img = xy_img[:, None, :, None, :] + offsets[None, None, None, :, :].expand(batch_size, 3, 1,
                                                                                      offsets.shape[0],
                                                                                      offsets.shape[1]).contiguous()
        xy_img_idx = xy_img.view(-1, 2)
        CHANEl_IDX = torch.arange(3)[None, :, None, None].expand(batch_size, 3, problem_size,
                                                                 offsets.shape[0]).contiguous().view(-1)
        BATCH_IDX = torch.arange(batch_size)[:, None, None, None].expand(batch_size, 3, problem_size,
                                                                         offsets.shape[0]).contiguous().view(-1)

        demand = torch.zeros((batch_size, 3, problem_size, offsets.shape[0], 1))
        ori_value = (td["demand_linehaul"] + td["demand_backhaul"])

        max_v = ori_value[:,1:].max(dim=-1).values[:, None]
        ori_value[ori_value == 0] = 1
        min_v = ori_value[:,1:].min(dim=-1).values[:, None]
        min_v = torch.where(min_v == max_v, 0, min_v)
        dli = ((td["demand_linehaul"] - min_v) / (max_v - min_v)).unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                     problem_size,
                                                                                                     offsets.shape[0],
                                                                                                     1)
        db = ((td["demand_backhaul"] - min_v) / (max_v - min_v)).unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                    problem_size,
                                                                                                    offsets.shape[
                                                                                                        0], 1)

        demand[:, 0, :, :, :] = torch.zeros((batch_size, problem_size, offsets.shape[0], 1))
        demand[:, 1, :, :, :] = dli
        demand[:, 2, :, :, :] = db

        demand[:, 0, 0, :, :] = torch.ones((batch_size, offsets.shape[0], 1))
        demand[:, 1, 0, :, :] = torch.zeros((batch_size, offsets.shape[0], 1))
        demand[:, 2, 0, :, :] = torch.zeros((batch_size, offsets.shape[0], 1))

        dl = torch.nan_to_num(td["distance_limit"], posinf=self.posinf_val)

        idx_m = torch.where(dl > 0)
        BATCH_IDX_M = idx_m[0][:, None].expand(len(idx_m[0]), offsets_m.shape[0]).contiguous().view(-1)
        # NODE_IDX_M = idx_m[1][:, None].expand(len(idx_m[0]), offsets_m.shape[0]).contiguous().view(-1)
        PIXEL_IDX_M = offsets_m[None, :].expand(len(idx_m[0]), offsets_m.shape[0]).contiguous().view(-1)
        demand[BATCH_IDX_M, :, :, PIXEL_IDX_M, :] = bg[BATCH_IDX_M][:, None, None, :].expand(BATCH_IDX_M.shape[0], 3,
                                                                                             problem_size, 1)

        demand = demand.contiguous().view(-1)

        imgs[BATCH_IDX, CHANEl_IDX, xy_img_idx[:, 0], xy_img_idx[:, 1]] = demand
        torch.set_default_tensor_type('torch.FloatTensor')

        return imgs, final_block_indices, final_block_indices_l

    def get_problems_img_LTW(self, batch_size, problem_size, td):
        problem_size = problem_size + 1
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        problems = td["locs"]
        img_size = 224

        if problem_size > 100:
            offsets = torch.tensor([[0, 0], [0, 1], [0, -1],
                                    [1, -1], [1, 0], [1, 1],
                                    [-1, -1], [-1, 0], [-1, 1]])
            offsets_m = torch.tensor([3, 5,
                                      6, 8])
        else:
            offsets = torch.tensor([[0, 0], [0, 1], [0, 2], [0, -2], [0, -1],
                                    [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                    [2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
                                    [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
                                    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2]])
            offsets_m = torch.tensor([5, 6, 8, 9,
                                      10, 11, 13, 14,
                                      15, 16, 18, 19,
                                      20, 21, 23, 24])
        bg = td["open_route"].float()
        bg[bg == 0] = 0.25
        bg[bg == 1] = 0.75
        imgs = torch.ones((batch_size, 3, img_size, img_size)) * bg.unsqueeze(-1).unsqueeze(-1)
        xy_img = problems * 200 + 12
        xy_img = xy_img.int()
        xy_img = xy_img[:, None, :, None, :] + offsets[None, None, None, :, :].expand(batch_size, 3, 1,
                                                                                      offsets.shape[0],
                                                                                      offsets.shape[1]).contiguous()
        xy_img_idx = xy_img.view(-1, 2)
        CHANEl_IDX = torch.arange(3)[None, :, None, None].expand(batch_size, 3, problem_size,
                                                                 offsets.shape[0]).contiguous().view(-1)
        BATCH_IDX = torch.arange(batch_size)[:, None, None, None].expand(batch_size, 3, problem_size,
                                                                         offsets.shape[0]).contiguous().view(-1)

        demand = torch.zeros((batch_size, 3, problem_size, offsets.shape[0], 1))
        tw = torch.nan_to_num(td["time_windows"], posinf=self.posinf_val)

        st = tw[:, :, 0] + td["service_time"]
        if st.max() > st.min():
            st = (st - st.min()) / (st.max() - st.min())
        r = (255 * st + (1 - st) * 255) / 255
        g = (165 * st + (1 - st) * 255) / 255
        b = (0 * st + (1 - st) * 255) / 255

        st_r = r.unsqueeze(-1).unsqueeze(-1).expand(batch_size, problem_size, offsets.shape[0], 1)
        st_g = g.unsqueeze(-1).unsqueeze(-1).expand(batch_size, problem_size, offsets.shape[0], 1)
        st_b = b.unsqueeze(-1).unsqueeze(-1).expand(batch_size, problem_size, offsets.shape[0], 1)

        demand[:, 0, :, :, :] = st_r
        demand[:, 1, :, :, :] = st_g
        demand[:, 2, :, :, :] = st_b

        demand[:, 0, 0, :, :] = torch.ones((batch_size, offsets.shape[0], 1))
        demand[:, 1, 0, :, :] = torch.zeros((batch_size, offsets.shape[0], 1))
        demand[:, 2, 0, :, :] = torch.zeros((batch_size, offsets.shape[0], 1))

        dl = torch.nan_to_num(td["distance_limit"], posinf=self.posinf_val)

        idx_m = torch.where(dl > 0)
        BATCH_IDX_M = idx_m[0][:, None].expand(len(idx_m[0]), offsets_m.shape[0]).contiguous().view(-1)
        # NODE_IDX_M = idx_m[1][:, None].expand(len(idx_m[0]), offsets_m.shape[0]).contiguous().view(-1)
        PIXEL_IDX_M = offsets_m[None, :].expand(len(idx_m[0]), offsets_m.shape[0]).contiguous().view(-1)
        demand[BATCH_IDX_M, :, :, PIXEL_IDX_M, :] = bg[BATCH_IDX_M][:, None, None, :].expand(BATCH_IDX_M.shape[0], 3,
                                                                                             problem_size, 1)

        demand = demand.contiguous().view(-1)

        imgs[BATCH_IDX, CHANEl_IDX, xy_img_idx[:, 0], xy_img_idx[:, 1]] = demand
        torch.set_default_tensor_type('torch.FloatTensor')

        return imgs


class MTVRPInitEmbeddingGlobalNoPolar(MTVRPInitEmbeddingRouteFinderBase):
    """
    Also global embedding, but without polar coordinates. Here we also project the depot position.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingGlobalNoPolar, self).__init__(
            num_global_feats=4,
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
            ],
            -1,
        )

    def _cust_feats(self, td):
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )


## NOTE: we should use *this*!
class MTVRPInitEmbeddingGlobalPolar(MTVRPInitEmbeddingRouteFinderBase):
    """This version transforms to polar coordinates first and also uses our global embedding features.
    Global features:
        - open_route (O)
        - distance_limit (L)
    The above features are embedded in the depot node as global and get broadcasted via attention.
    This allows the network to learn the relationships between them.

    Note that we don't include the depot position here since it is always 0! (centered) because of our transformation to polar coordinates.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingGlobalPolar, self).__init__(
            num_global_feats=2,
            num_cust_feats=9,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["distance_limit"][..., None],
            ],
            -1,
        )

    def _cust_feats(self, td):
        locs = td["locs"]
        depot = locs[:, 0:1, :]
        locs = locs - depot  # centering

        dist_to_depot = torch.norm(locs, p=2, dim=-1, keepdim=True)
        angle_to_depot = torch.atan2(locs[..., 1:], locs[..., :1])

        return torch.cat(
            (
                dist_to_depot[..., 1:, :],
                angle_to_depot[..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                locs[:, 1:, :],
            ),
            -1,
        )


class MTVRPInitEmbeddingRouteFinder(MTVRPInitEmbeddingGlobalPolar):
    def __init__(self, *args, **kwargs):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(*args, **kwargs)


# Simple ablation without the location features
class MTVRPInitEmbeddingGlobalPolarOnly(MTVRPInitEmbeddingGlobalPolar):
    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingGlobalPolar, self).__init__(
            num_global_feats=2,
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _cust_feats(self, td):
        locs = td["locs"]
        depot = locs[:, 0:1, :]
        locs = locs - depot  # centering

        dist_to_depot = torch.norm(locs - depot, p=2, dim=-1, keepdim=True)
        angle_to_depot = torch.atan2(locs[..., 1:], locs[..., :1])

        return torch.cat(
            (
                dist_to_depot[..., 1:, :],
                angle_to_depot[..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )


class ZeroShotInitEmbedding(MTVRPInitEmbeddingRouteFinder):
    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        # Note: here we add the backhaul_class as a feature
        MTVRPInitEmbeddingRouteFinderBase.__init__(
            self,
            num_global_feats=3,
            num_cust_feats=9,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        glob_feats = super(ZeroShotInitEmbedding, self)._global_feats(td)
        is_mixed_backhaul = (td["backhaul_class"] == 2).float()
        return torch.cat([glob_feats, is_mixed_backhaul[..., None]], -1)

    # def get_problems_img(self, batch_size, problem_size, td):
    #     problem_size = problem_size + 1
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     problems = td["locs"]
    #     patches = 7
    #     patches_l = 28
    #     img_size = 224
    #     patch_size = img_size // patches
    #     patch_size_l = img_size // patches_l
    #
    #     if problem_size > 100:
    #         offsets = torch.tensor([[0, 0], [0, 1], [0, -1],
    #                                 [1, -1], [1, 0], [1, 1],
    #                                 [-1, -1], [-1, 0], [-1, 1]])
    #         offsets_m = torch.tensor([[1, -1], [1, 1],
    #                                   [-1, -1], [-1, 1]])
    #     else:
    #         offsets = torch.tensor([[0, 0], [0, 1], [0, 2], [0, -2], [0, -1],
    #                                 [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
    #                                 [2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
    #                                 [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
    #                                 [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2]])
    #         offsets_m = torch.tensor([[1, -2], [1, -1], [1, 1], [1, 2],
    #                                   [2, -2], [2, -1], [2, 1], [2, 2],
    #                                   [-2, -2], [-2, -1], [-2, 1], [-2, 2],
    #                                   [-1, -2], [-1, -1], [-1, 1], [-1, 2]])
    #     bg = td["open_route"].float()
    #     imgs = torch.ones((batch_size, 3, img_size, img_size)) * bg.unsqueeze(-1).unsqueeze(-1)
    #     xy_img = problems * 200 + 12
    #     xy_img = xy_img.int()
    #     block_indices = xy_img // patch_size
    #     final_block_indices = block_indices[:, :, 0] * patches + block_indices[:, :, 1]
    #     block_indices_l = xy_img // patch_size_l
    #     final_block_indices_l = block_indices_l[:, :, 0] * patches + block_indices_l[:, :, 1]
    #     xy_img_ori = xy_img[:, None, :, None, :] + offsets[None, None, None, :, :].expand(batch_size, 3, 1,
    #                                                                                       offsets.shape[0],
    #                                                                                       offsets.shape[1]).contiguous()
    #     xy_img_idx = xy_img_ori.view(-1, 2)
    #     is_mixed_backhaul = (td["backhaul_class"] == 2).float()
    #     idx_m = torch.where(td["backhaul_class"] == 2)
    #     if len(idx_m[0]) > 0:
    #         xy_img_m = xy_img[idx_m[0], None, :, None, :] + offsets_m[None, None, None, :, :].expand(len(idx_m[0]), 3,
    #                                                                                                  1,
    #                                                                                                  offsets_m.shape[0],
    #                                                                                                  offsets_m.shape[
    #                                                                                                      1]).contiguous()
    #     CHANEl_IDX = torch.arange(3)[None, :, None, None].expand(batch_size, 3, problem_size,
    #                                                              offsets.shape[0]).contiguous().view(-1)
    #     BATCH_IDX = torch.arange(batch_size)[:, None, None, None].expand(batch_size, 3, problem_size,
    #                                                                      offsets.shape[0]).contiguous().view(-1)
    #
    #     demand = torch.zeros((batch_size, 3, problem_size, offsets.shape[0], 1))
    #     ori_value = (td["demand_linehaul"] + td["demand_backhaul"])
    #
    #     max_v = ori_value.max(dim=-1).values[:, None]
    #     ori_value[ori_value == 0] = 1
    #     min_v = ori_value.min(dim=-1).values[:, None]
    #     dli = ((td["demand_linehaul"] - min_v) / (max_v - min_v)).unsqueeze(-1).unsqueeze(-1).expand(batch_size,
    #                                                                                                  problem_size,
    #                                                                                                  offsets.shape[0],
    #                                                                                                  1)
    #     db = ((td["demand_backhaul"] - min_v) / (max_v - min_v)).unsqueeze(-1).unsqueeze(-1).expand(batch_size,
    #                                                                                                 problem_size,
    #                                                                                                 offsets.shape[
    #                                                                                                     0], 1)
    #
    #     demand[:, 0, :, :, :] = torch.zeros((batch_size, problem_size, offsets.shape[0], 1))
    #     demand[:, 1, :, :, :] = dli
    #     demand[:, 2, :, :, :] = db
    #
    #     demand[:, 0, 0, :, :] = torch.ones((batch_size, offsets.shape[0], 1))
    #     demand[:, 1, 0, :, :] = torch.zeros((batch_size, offsets.shape[0], 1))
    #     demand[:, 2, 0, :, :] = torch.zeros((batch_size, offsets.shape[0], 1))
    #
    #     demand = demand.contiguous().view(-1)
    #
    #     imgs[BATCH_IDX, CHANEl_IDX, xy_img_idx[:, 0], xy_img_idx[:, 1]] = demand
    #     torch.set_default_tensor_type('torch.FloatTensor')
    #
    #     return imgs, final_block_indices, final_block_indices_l
