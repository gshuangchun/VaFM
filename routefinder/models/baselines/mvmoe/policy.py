from typing import Callable

import torch.nn as nn

from rl4co.models.zoo.am import AttentionModelPolicy
from fvcore.nn import FlopCountAnalysis
from .decoder import MVMoEDecoder
from .encoder import MVMoEEncoder
from rl4co.utils.ops import batchify
import abc
import torch
from typing import Any, Callable, Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MVMoEPolicy(AttentionModelPolicy):
    """
    https://github.com/RoyalSkye/Routing-MVMo
    """

    def __init__(
            self,
            encoder: nn.Module = None,
            decoder: nn.Module = None,
            embed_dim: int = 128,
            num_encoder_layers: int = 6,
            num_heads: int = 8,
            normalization: str = "instance",
            feedforward_hidden: int = 512,
            env_name: str = "mtvrp",
            encoder_network: nn.Module = None,
            init_embedding: nn.Module = None,
            context_embedding: nn.Module = None,
            dynamic_embedding: nn.Module = None,
            use_graph_context: bool = False,
            linear_bias_decoder: bool = False,
            sdpa_fn: Callable = None,
            mask_inner: bool = True,
            out_bias_pointer_attn: bool = False,
            check_nan: bool = True,
            temperature: float = 1.0,
            tanh_clipping: float = 10.0,
            mask_logits: bool = True,
            train_decode_type: str = "sampling",
            val_decode_type: str = "greedy",
            test_decode_type: str = "greedy",
            # MoE specific
            num_experts=4,
            routing_method="input_choice",
            routing_level="node",
            topk=2,
            moe_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
            hierarchical_gating=False,  # if True, corresponds to MVMoE-L
            **unused_kwargs,
    ):
        if encoder is None:
            encoder = MVMoEEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn,
                num_experts=num_experts,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
                moe_loc=moe_loc,
            )
        if decoder is None:
            decoder_NTW = MVMoEDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                num_experts=num_experts,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
                moe_loc=moe_loc,
                hierarchical_gating=hierarchical_gating,
            )
            decoder_TW = MVMoEDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                num_experts=num_experts,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
                moe_loc=moe_loc,
                hierarchical_gating=hierarchical_gating,
            )

        super(AttentionModelPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder_NTW,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
        self.decoder_TW = decoder_TW

        # decoders = [decoder_NTW, decoder_TW]

    def forward(
            self,
            td: TensorDict,
            env: Union[str, RL4COEnvBase] = None,
            phase: str = "train",
            calc_reward: bool = True,
            return_actions: bool = False,
            return_entropy: bool = False,
            return_init_embeds: bool = False,
            return_sum_log_likelihood: bool = True,
            actions=None,
            max_steps=1_000_000,
            **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # A = self.encoder.net.idx_NTW
        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        # offsets = torch.arange(num_starts).unsqueeze(1) * (td.shape[0] // num_starts)
        # idx_NTW = self.encoder.net.idx_NTW + offsets.cuda()
        # idx_NTW = idx_NTW.flatten()

        # Additionally call a decoder hook if needed before main decoding
        td_NTW, env_NTW, hidden_NTW = self.decoder.pre_decoder_hook(td, env, hidden[1], num_starts)
        td_TW, env_TW, hidden_TW = self.decoder_TW.pre_decoder_hook(td, env, hidden[0], num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits_NTW, mask_NTW = self.decoder(td_NTW, hidden_NTW, num_starts)
            logits_TW, mask = self.decoder_TW(td_TW, hidden_TW, num_starts)
            assert torch.equal(mask_NTW, mask)
            logits = torch.stack([logits_TW, logits_NTW], dim=0).mean(dim=0)
            # logits = logits_NTW
            # logits[idx_NTW] = logits_NTW[idx_NTW]
            # mask[idx_NTW] = mask_NTW[idx_NTW]
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict


class MVMoELightPolicy(MVMoEPolicy):
    def __init__(self, *args, **kwargs):
        # assert hierarchical_gating is set to true
        if "hierarchical_gating" in kwargs:
            assert kwargs[
                "hierarchical_gating"
            ], "hierarchical_gating must be set to True for MVMoELPolicy"

        kwargs["hierarchical_gating"] = True

        super(MVMoELightPolicy, self).__init__(*args, **kwargs)
