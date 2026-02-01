import random

from typing import Any, Union
from torch import ScriptModule, Tensor
import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RouteFinderBase(POMO):
    """
    Main RouteFinder RL model based on POMO.
    This automatically include the Mixed Batch Training (MBT) from the environment.
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module,
            **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        # Initialize with the shared baseline
        super(RouteFinderBase, self).__init__(env, policy, **kwargs)


class RouteFinderMoE(RouteFinderBase):
    """RouteFinder with MoE model as the policy as in MVMoE (https://github.com/RoyalSkye/Routing-MVMoE).
    This includes the Mixed Batch Training (MBT) from the environment.
    Note that additional losses are added to the loss function for MoE during training.
    Note that to use the new embeddings, you should pass them to the new policy via:
    - init_embedding: MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
    - context_embedding: MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)
    
    Ref for MVMoE: 
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module,
            **kwargs,
    ):

        from routefinder.models.baselines.mvmoe.policy import MVMoELightPolicy, MVMoEPolicy

        assert isinstance(
            policy, (MVMoEPolicy, MVMoELightPolicy)
        ), "policy must be an instance of MVMoEPolicy or MVMoELightPolicy"

        super(RouteFinderMoE, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
            self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # Shared step from POMO
        out = super(RouteFinderMoE, self).shared_step(
            batch, batch_idx, phase, dataloader_idx
        )

        # Get loss
        loss = out.get("loss", None)

        if loss is not None:
            # Init embeddings
            # Option 1 in the code
            if hasattr(self.policy.encoder.init_embedding, "moe_loss"):
                moe_loss_init_embeds = self.policy.encoder.init_embedding.moe_loss
            else:
                moe_loss_init_embeds = 0

            # Encoder layers
            # Option 2 in the code
            moe_loss_layers = 0
            for layer in self.policy.encoder.net.layers:
                if hasattr(layer, "moe_loss"):
                    moe_loss_layers = moe_loss_layers + layer.moe_loss
                else:
                    moe_loss_layers = moe_loss_layers + 0

            # Decoder layer
            # Option 3 in the code
            if hasattr(self.policy.decoder.pointer, "moe_loss"):
                moe_loss_decoder = self.policy.decoder.pointer.moe_loss
            else:
                moe_loss_decoder = 0

            # Sum losses and save in out for backpropagation
            moe_loss = moe_loss_init_embeds + moe_loss_layers + moe_loss_decoder
            # img_loss = self.policy.encoder.net.loss_mse
            img_loss_cls = self.policy.encoder.net.loss_cls
            # img_loss_proto = self.policy.encoder.net.loss_proto
            out["loss"] = loss + moe_loss + 0.1 * img_loss_cls
            # out["loss_mse"] = img_loss
            # out["loss_proto"] = img_loss_proto
            out["loss_cls"] = img_loss_cls
            out["loss_ori"] = loss + moe_loss
            out["loss_coor"] = loss
            out["loss_moe"] = moe_loss
        _ = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)

        return out

    def log_metrics(self, metric_dict: dict, phase: str, dataloader_idx: int = None):
        """Log metrics to logger and progress bar"""
        metrics = getattr(self, f"{phase}_metrics")
        dataloader_name = ""
        if dataloader_idx is not None and self.dataloader_names is not None:
            dataloader_name = "/" + self.dataloader_names[dataloader_idx]
        metrics = {
            f"{phase}/{k}{dataloader_name}": v.mean()
            if isinstance(v, torch.Tensor)
            else v
            for k, v in metric_dict.items()
            if k in metrics
        }
        log_on_step = self.log_on_step if phase == "train" else False
        on_epoch = False if phase == "train" else True
        if phase == 'train' and "loss_moe" not in metric_dict.keys():
            return metrics
        else:
            self.log_dict(
                metrics,
                on_step=log_on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                sync_dist=True,
                add_dataloader_idx=False,  # we add manually above
            )
            return metrics

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).

        Example::

            def backward(self, loss):
                loss.backward()

        """
        loss.backward(retain_graph=True)


class RouteFinderSingleVariantSampling(RouteFinderBase):
    """This is the default sampling method for MVMoE and MTPOMO
    (without Mixed-Batch Training) as first proposed in MTPOMO (https://arxiv.org/abs/2402.16891) 
    
    The environment generates by default all the features,
    and we subsample them at each batch to train the model (i.e. we select a subset of the features).
    
    For example: we always sample OVRPBLTW (with all features) and we simply take a subset of them at each batch.
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module,
            preset: str = "all",  # all or single_feat_otw
            **kwargs,
    ):
        # assert that the env generator has all the features
        assert (
                env.generator.variant_preset == "all"
        ), "The env generator must have all the features since we are sampling them"
        assert preset in [
            "all",
            "single_feat_otw",
        ], "preset must be either all or single_feat_otw"
        self.preset = preset
        assert (
                env.generator.subsample == False
        ), "The env generator must not subsample the features, this is done in the `shared_step` method"

        super(RouteFinderSingleVariantSampling, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
            self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = batch

        # variant subsampling: given a batch with *all* features, we subsample a part of them
        if phase == "train":
            # OTW: CVRP, VRPTW, VRPL, VRPB, OVRP, OVRPTW
            if self.preset == "single_feat_otw":
                # Sample single feature
                indices_idx = random.randint(0, 5)
                indices = [1 if i != indices_idx else 0 for i in range(5)]

                # incides_idx == 4 -> "OTW"
                if indices_idx == 4:
                    indices[0] = 0
                    indices[1] = 0

            # All features; we select randomly a subset of the features
            elif self.preset == "all":
                # Sample single variant (i.e which features to *remove* with 50% probability)
                indices = torch.bernoulli(torch.tensor([0.5] * 4))

            # Process the indices
            if indices[0] == 1:  # Remove open
                td["open_route"] &= False
            if indices[1] == 1:  # Remove time window
                td["time_windows"][..., 0] = td["time_windows"][..., 0] + 0
                td["time_windows"][..., 1] = td["time_windows"][..., 1] + float("inf")
                td["service_time"] *= 0
            if indices[2] == 1:  # Remove distance limit
                td["distance_limit"] = td["distance_limit"] + float("inf")
            if indices[3] == 1:  # Remove backhaul
                td.set("demand_linehaul", td["demand_linehaul"] + td["demand_backhaul"])
                td.set("demand_backhaul", torch.zeros_like(td["demand_backhaul"]))

        return super(RouteFinderSingleVariantSampling, self).shared_step(
            td, batch_idx, phase, dataloader_idx
        )
