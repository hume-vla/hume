import math
from copy import deepcopy
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    TanhTransform,
)
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
)
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from .. import array_typing as at


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class VQHBackboneConfig(PretrainedConfig):
    model_type = "VQHBackbone"
    sub_configs = {"gemma_expert_config": AutoConfig}

    def __init__(
        self,
        gemma_expert_config: dict | None = None,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.attention_implementation = attention_implementation

        if gemma_expert_config is None:
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=4,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(gemma_expert_config, dict):
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"
            cfg_cls = CONFIG_MAPPING[gemma_expert_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(
        torch.float32
    )

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class VQHBackbone(PreTrainedModel):
    config_class = VQHBackboneConfig

    def __init__(self, config: VQHBackboneConfig):
        super().__init__(config=config)
        self.config = config
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)

        self.to_bfloat16_like_physical_intelligence()

    def train(self, mode: bool = True):
        super().train(mode)

    def to_bfloat16_like_physical_intelligence(self):
        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        # RMSNorm
        head_dim = self.gemma_expert.config.head_dim

        hidden_states = inputs_embeds
        batch_size = hidden_states.shape[0]
        for layer in self.gemma_expert.model.layers[
            : self.gemma_expert.config.num_hidden_layers
        ]:
            # normalizer = torch.tensor(model.config.hidden_size**0.5, dtype=hidden_states.dtype)
            # hidden_states = hidden_states * normalizer
            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            # self attention
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
            query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask,
                batch_size,
                head_dim,
                query_states,
                key_states,
                value_states,
            )

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

            out_emb = layer.self_attn.o_proj(att_output)

            # first residual
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            out_emb = layer.post_attention_layernorm(out_emb)
            out_emb = layer.mlp(out_emb)
            # second residual
            out_emb += after_first_residual
            hidden_states = out_emb

        # final norm
        hidden_states = self.gemma_expert.model.norm(hidden_states)

        return hidden_states

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
    ):
        num_att_heads = self.config.gemma_expert_config.num_attention_heads
        num_key_value_heads = self.config.gemma_expert_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(
            attention_mask[:, None, :, :], att_weights, big_neg
        )

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output


class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Tuple[int, ...] = (),
        constraint_type: str = "eq",  # One of ("eq", "leq", "geq")
        parameterization: Optional[
            str
        ] = None,  # One of ("softplus", "exp"), or None for equality constraints
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization

        if constraint_type != "eq":
            assert (
                init_value > 0
            ), "Inequality constraints must have non-negative initial multiplier values"

            if parameterization == "softplus":
                init_value = torch.log(torch.exp(torch.tensor(init_value)) - 1).item()
            elif parameterization == "exp":
                init_value = torch.log(torch.tensor(init_value)).item()
            else:
                raise ValueError(
                    f"Invalid multiplier parameterization {parameterization}"
                )
        else:
            assert (
                parameterization is None
            ), "Equality constraints must have no parameterization"

        self.multiplier = nn.Parameter(torch.full(constraint_shape, init_value))

    def forward(
        self, lhs: Optional[torch.Tensor] = None, rhs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        multiplier = self.multiplier

        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = torch.nn.functional.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = torch.exp(multiplier)
            else:
                raise ValueError(
                    f"Invalid multiplier parameterization {self.parameterization}"
                )

        if lhs is None:
            return multiplier

        if rhs is None:
            rhs = torch.zeros_like(lhs)

        diff = lhs - rhs

        assert (
            diff.shape == multiplier.shape
        ), f"Shape mismatch: {diff.shape} vs {multiplier.shape}"

        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff


GeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="geq", parameterization="softplus"
)

LeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="leq", parameterization="softplus"
)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activations: Union[Callable[[torch.Tensor], torch.Tensor], str] = "silu",
        activate_final: bool = False,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()

        assert not (use_layer_norm and use_group_norm)

        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        if isinstance(activations, str):
            if activations == "silu" or activations == "swish":
                self.activations = nn.SiLU()
            else:
                self.activations = getattr(nn, activations)()
        else:
            self.activations = activations

        layers = []

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

            input_dim = hidden_dim

            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))

                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                elif use_group_norm:
                    num_groups = min(hidden_dim, 32)
                    layers.append(nn.GroupNorm(num_groups, hidden_dim))
                layers.append(self.activations)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        self.loc = loc
        self.scale_diag = scale_diag
        base_distribution = Independent(Normal(loc, scale_diag), 1)

        transforms = []
        transforms.append(TanhTransform())
        if not (low is None or high is None):
            transforms.append(
                AffineTransform(loc=(high + low) / 2, scale=(high - low) / 2)
            )
        transform = ComposeTransform(transforms)

        super().__init__(base_distribution, transform)

    def mode(self) -> torch.Tensor:
        mode = self.loc
        for transform in self.transforms:
            mode = transform(mode)
        return mode

    def stddev(self) -> torch.Tensor:
        return self.transform(self.loc + self.scale_diag) - self.transform(self.loc)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        value = torch.clamp(value, -1 + eps, 1 - eps)
        return super().log_prob(value)


class Policy(nn.Module):
    def __init__(
        self,
        obs_encoded_dim: int,
        network: nn.Module,
        action_dim: int,
        std_parameterization: str = "exp",  # "exp", "softplus", "fixed", or "uniform"
        std_min: Optional[float] = 1e-5,
        std_max: Optional[float] = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.obs_encoded_dim = obs_encoded_dim
        self.network = network
        self.action_dim = action_dim
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std

        self.mean_layer = nn.Linear(network.hidden_dims[-1], action_dim)

        if fixed_std is None:
            if std_parameterization in ["exp", "softplus"]:
                self.std_layer = nn.Linear(network.hidden_dims[-1], action_dim)
            elif std_parameterization == "uniform":
                self.log_stds = nn.Parameter(torch.zeros(action_dim))
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert std_parameterization == "fixed"

        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)

        if fixed_std is None and std_parameterization in ["exp", "softplus"]:
            nn.init.xavier_uniform_(self.std_layer.weight)
            nn.init.zeros_(self.std_layer.bias)

    def forward(
        self, encoded_observations: torch.Tensor, temperature: float = 1.0
    ) -> Union[TransformedDistribution, Normal]:
        outputs = self.network(encoded_observations)

        means = self.mean_layer(outputs)

        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(outputs)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = self.std_layer(outputs)
                stds = nn.functional.softplus(stds)
            elif self.std_parameterization == "uniform":
                stds = torch.exp(self.log_stds).expand_as(means)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            stds = self.fixed_std.to(means.device).expand_as(means)

        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(
            torch.tensor(temperature)
        )

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = Normal(loc=means, scale=stds)

        return distribution


class Critics(nn.Module):
    def __init__(
        self,
        obs_encoded_dim: int,
        networks: list[nn.Module],
        num_backbones: int = 2,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        assert len(networks) == num_backbones
        self.obs_encoded_dim = obs_encoded_dim
        self.networks = nn.ModuleList(networks)
        self.num_backbones = num_backbones
        self.init_final = init_final

        self.backbone_output_dims = networks[0].hidden_dims[-1]

        if init_final is not None:
            self.output_layer = nn.Linear(self.backbone_output_dims, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(self.backbone_output_dims, 1)
            nn.init.xavier_uniform_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        encoded_observations: Float[torch.Tensor, "batch {self.obs_encoded_dim}"],
        actions: Float[torch.Tensor, "batch *num_actions action_dim"],
    ) -> Float[torch.Tensor, "{self.num_backbones} batch *num_actions"]:
        if actions.ndim == 3:
            # forward the q function with multiple actions on each state
            encoded_observations = encoded_observations.unsqueeze(1).expand(
                -1, actions.shape[1], -1
            )
        # HACK: check dimensions here
        inputs = torch.cat([encoded_observations, actions], dim=-1)

        backbone_outputs = []
        for network in self.networks:
            backbone_outputs.append(network(inputs))
        backbone_outputs: Float[
            torch.Tensor,
            "{self.num_backbones} batch *num_actions {self.backbone_output_dims}",
        ] = torch.stack(backbone_outputs, dim=0)

        value = self.output_layer(backbone_outputs)
        # HACK: check output shape here
        # if actions.ndim == 3:
        #     value = value.squeeze(-1).permute(0, 2, 1)
        # else:
        value = value.squeeze(-1)
        return value  # (num_backbones, batch, *num_actions)


class CalQlConfig(PretrainedConfig):
    moedel_type = "calql"

    def __init__(
        self,
        obs_encoded_dim=2048,
        action_dim=70,
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        actor_wps=2000,
        critic_wps=2000,
        **kwargs,
    ):
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        self.cql_alpha = 5.0
        self.cql_autotune_alpha = False
        self.action_dim = action_dim
        self.target_entropy = -self.action_dim
        self.obs_encoded_dim = obs_encoded_dim
        self.cql_temperature_init_value = 1.0
        self.critic_ensemble_size = 2
        self.cql_n_actions = 4
        self.cql_max_target_backup = True
        self.policy_network_kwargs = dict(
            input_dim=self.obs_encoded_dim,
            hidden_dims=[256, 256],
            activate_final=True,
            use_layer_norm=False,
        )
        self.critic_network_kwargs = dict(
            input_dim=self.obs_encoded_dim + self.action_dim,
            hidden_dims=[256, 256],
            activate_final=True,
            use_layer_norm=False,
        )
        self.policy_kwargs = dict(
            tanh_squash_distribution=True,
            std_parameterization="exp",
        )
        self.critic_subsample_size = None
        self.cql_max_target_backup = True
        self.backup_entropy = False
        self.discount = 0.98
        self.goal_conditioned = True
        self.gc_kwargs = dict(
            negative_proportion=0.0,
        )
        self.use_td_loss = True
        self.cql_action_sample_method = "uniform"
        self.cql_importance_sample = True
        self.cql_temp = 1.0
        self.use_calql = True

        self.actor_optimizer_kwargs = dict(
            learning_rate=actor_lr,
            warmup_steps=actor_wps,
        )
        self.critic_optimizer_kwargs = dict(
            learning_rate=critic_lr,
            warmup_steps=critic_wps,
        )
        self.temperature_optimizer_kwargs = dict(learning_rate=temp_lr)

        super().__init__(**kwargs)


class CalQL(PreTrainedModel):
    config_calss = CalQlConfig

    def __init__(self, config: CalQlConfig):
        super(CalQL, self).__init__(config=config)
        self.config = config

        self.temperature = GeqLagrangeMultiplier(
            init_value=self.config.cql_temperature_init_value,
            constraint_shape=(),
        )

        self.policy = Policy(
            obs_encoded_dim=self.config.obs_encoded_dim,
            network=MLP(**self.config.policy_network_kwargs),
            action_dim=self.config.action_dim,
            **self.config.policy_kwargs,
        )

        self.critics = Critics(
            obs_encoded_dim=self.config.obs_encoded_dim,
            networks=[
                MLP(**self.config.critic_network_kwargs)
                for _ in range(self.config.critic_ensemble_size)
            ],
            num_backbones=self.config.critic_ensemble_size,
        )

        self.target_critics = deepcopy(self.critics)

    def forward_policy_and_sample(
        self,
        encoded_obs: Float[torch.Tensor, "batch {self.config.obs_encoded_dim}"],
        repeat: int = None,
    ):
        action_dist = self.policy.forward(encoded_obs)
        if repeat:
            new_actions = action_dist.rsample(
                torch.tensor([repeat])
            )  # repeat, tensor, act_dim
            log_pi = action_dist.log_prob(new_actions)
            new_actions = new_actions.permute(1, 0, 2)  # (batch, repeat, action_dim)
            log_pi = log_pi.permute(1, 0)  # (batch, repeat)

        else:
            new_actions = action_dist.rsample()  # (batch, action_dim)
            log_pi = action_dist.log_prob(new_actions)  # (batch)
        # NOTE: detach gradient here
        new_actions = new_actions.detach()
        log_pi = log_pi.detach()
        return new_actions, log_pi

    def _compute_next_actions(self, batch: at.CalQlBatch):
        """
        compute the next actions but with repeat cql_n_actions times
        this should only be used when calculating critic loss using
        cql_max_target_backup
        """
        sample_n_actions = (
            self.config.cql_n_actions if self.config.cql_max_target_backup else None
        )

        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["encoded_next_observations"],
            repeat=sample_n_actions,
        )
        return next_actions, next_actions_log_probs

    def temperature_loss_fn(self, batch: at.CalQlBatch):
        next_actions, next_actions_log_probs = self._compute_next_actions(batch)

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature.forward(
            lhs=entropy,
            rhs=self.config.target_entropy,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    def policy_loss_fn(self, batch: at.CalQlBatch):
        batch_size = batch["rewards"].shape[0]
        temperature = self.temperature.forward().detach()  # detach gradient

        action_distributions = self.policy.forward(batch["encoded_observations"])
        actions = action_distributions.rsample()
        log_probs = action_distributions.log_prob(actions)

        predicted_qs = self.critics.forward(
            batch["encoded_observations"],
            actions,
        ).detach()  # NOTE: detach grads
        predicted_q = predicted_qs.min(dim=0)[0]

        assert predicted_q.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)

        nll_objective = -torch.mean(
            action_distributions.log_prob(torch.clip(batch["actions"], -0.99, 0.99))
        )
        actor_objective = predicted_q
        actor_loss = -torch.mean(actor_objective) + torch.mean(temperature * log_probs)

        info = {
            "actor_loss": actor_loss,
            "actor_nll": nll_objective,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((actions - batch["actions"]) ** 2).sum(dim=-1).mean(),
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
        }

        return actor_loss, info

    def sac_critic_loss_fn(self, batch: at.CalQlBatch):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        next_actions, next_actions_log_probs = self._compute_next_actions(batch)
        # (batch_size, ) for sac, (batch_size, cql_n_actions) for cql

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        with torch.no_grad():
            self.target_critics.eval()
            target_next_qs = self.target_critics.forward(
                batch["encoded_next_observations"],
                next_actions,
            )  # (critic_ensemble_size, batch_size, cql_n_actions)
            self.target_critics.train()

        # Subsample if requested
        if self.config.critic_subsample_size is not None:
            subsample_idcs = torch.randint(
                0,
                self.config.critic_ensemble_size,
                (self.config.critic_ensemble_size,),
                device=target_next_qs.device,
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(dim=0)[0]
        assert target_next_min_q.shape == next_actions_log_probs.shape
        # (batch_size,) for sac, (batch_size, cql_n_actions) for cql

        target_next_min_q = self._process_target_next_qs(
            target_next_min_q,
            next_actions_log_probs,
        )

        target_q = (
            batch["rewards"] + self.config.discount * batch["masks"] * target_next_min_q
        )
        assert target_q.shape == (batch_size,)

        predicted_qs = self.critics.forward(
            batch["encoded_observations"], batch["actions"]
        )
        assert predicted_qs.shape == (self.config.critic_ensemble_size, batch_size)

        target_qs = target_q.unsqueeze(0).expand(self.config.critic_ensemble_size, -1)
        assert predicted_qs.shape == target_qs.shape
        critic_loss = torch.mean((predicted_qs - target_qs) ** 2)

        info = {
            "td_err": critic_loss,
            "online_q": torch.mean(predicted_qs),
            "target_q": torch.mean(target_qs),
        }

        if self.config.goal_conditioned:
            num_negatives = int(
                self.config.gc_kwargs["negative_proportion"] * batch_size
            )
            info["negative_qs"] = torch.mean(predicted_qs, dim=-1)[
                :num_negatives
            ].mean()
            info["positive_qs"] = torch.mean(predicted_qs, dim=-1)[
                num_negatives:
            ].mean()

        return critic_loss, info

    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """add cql_max_target_backup option"""

        if self.config.cql_max_target_backup:
            max_target_indices = torch.argmax(target_next_qs, dim=-1, keepdim=True)
            target_next_qs = torch.gather(
                target_next_qs, -1, max_target_indices
            ).squeeze(-1)
            next_actions_log_probs = torch.gather(
                next_actions_log_probs, -1, max_target_indices
            ).squeeze(-1)

        target_next_qs = self.sac_process_target_next_qs(
            target_next_qs,
            next_actions_log_probs,
        )

        return target_next_qs

    def sac_process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """classes that inherit this class can add to this function
        e.g. CQL will add the cql_max_target_backup option
        """
        if self.config.backup_entropy:
            temperature = self.forward_temperature()
            target_next_qs = target_next_qs - temperature * next_actions_log_probs

        return target_next_qs

    def critic_loss_fn(self, batch: at.CalQlBatch):
        """add CQL loss on top of SAC loss"""
        if self.config.use_td_loss:
            td_loss, td_loss_info = self.sac_critic_loss_fn(batch)
        else:
            td_loss, td_loss_info = 0.0, {}

        cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(batch)

        """auto tune cql alpha"""
        if self.config.cql_autotune_alpha:
            raise NotImplementedError
            # alpha = self.forward_cql_alpha_lagrange()
            # cql_loss = (cql_q_diff - self.config["cql_target_action_gap"]).mean()
        else:
            alpha = self.config.cql_alpha
            cql_loss = torch.clip(
                cql_q_diff, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
            ).mean()

        critic_loss = td_loss + alpha * cql_loss

        info = {
            **td_loss_info,
            "critic_loss": critic_loss,
            "td_err": td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": alpha,
            "cql_diff": cql_q_diff.mean(),
            **cql_intermediate_results,
        }

        return critic_loss, info

    def _get_cql_q_diff(self, batch: at.CalQlBatch):
        """
        most of the CQL loss logic is here
        It is needed for both critic_loss_fn and cql_alpha_loss_fn
        """
        batch_size = batch["rewards"].shape[0]

        q_pred = self.critics.forward(batch["encoded_observations"], batch["actions"])
        # HACK: shape changed from jax implementation
        assert q_pred.shape == (self.config.critic_ensemble_size, batch_size)

        """sample random actions"""
        action_dim = batch["actions"].shape[-1]
        if self.config.cql_action_sample_method == "uniform":
            cql_random_actions = (
                torch.rand(
                    (batch_size, self.config.cql_n_actions, action_dim),
                    device=batch["actions"].device,
                )
                * 2.0
                - 1.0
            )
        elif self.config.cql_action_sample_method == "normal":
            cql_random_actions = torch.randn(
                (batch_size, self.config.cql_n_actions, action_dim),
                device=batch["actions"].device,
            )
        else:
            raise NotImplementedError

        cql_current_actions, cql_current_log_pis = self.forward_policy_and_sample(
            batch["encoded_observations"],
            repeat=self.config.cql_n_actions,
        )
        assert cql_current_log_pis.shape == (batch_size, self.config.cql_n_actions)

        cql_next_actions, cql_next_log_pis = self.forward_policy_and_sample(
            batch["encoded_next_observations"],
            repeat=self.config.cql_n_actions,
        )

        all_sampled_actions = torch.cat(
            [
                cql_random_actions,
                cql_current_actions,
                cql_next_actions,
            ],
            dim=1,
        )

        """q values of randomly sampled actions"""
        cql_q_samples = self.critics.forward(
            batch["encoded_observations"], all_sampled_actions
        )
        # HACK: shape changed from jax implementation
        assert cql_q_samples.shape == (
            self.config.critic_ensemble_size,
            batch_size,
            self.config.cql_n_actions * 3,
        )

        if self.config.critic_subsample_size is not None:
            subsample_idcs = torch.randint(
                0,
                self.config.critic_ensemble_size,
                (self.config.critic_ensemble_size,),
                device=cql_q_samples.device,
            )
            cql_q_samples = cql_q_samples[subsample_idcs]

        """Cal-QL"""
        if self.config.use_calql:
            # HACK: check shape of mc_returns
            mc_lower_bound = (
                batch["mc_returns"]
                .reshape(-1, 1)
                .repeat(1, self.config.cql_n_actions * 2)
            )
            assert mc_lower_bound.shape == (
                batch_size,
                self.config.cql_n_actions * 2,
            )

            cql_q_pi = cql_q_samples[:, :, self.config.cql_n_actions :]
            num_vals = cql_q_pi.numel()
            calql_bound_rate = torch.sum((cql_q_pi < mc_lower_bound).float()) / num_vals
            cql_q_pi = torch.maximum(cql_q_pi, mc_lower_bound)
            cql_q_samples = torch.cat(
                [
                    cql_q_samples[:, :, : self.config.cql_n_actions],
                    cql_q_pi,
                ],
                dim=-1,
            )

        if self.config.cql_importance_sample:
            random_density = torch.log(
                torch.tensor(0.5**action_dim, device=cql_q_samples.device)
            )

            importance_prob = torch.cat(
                [
                    random_density.expand(batch_size, self.config.cql_n_actions),
                    cql_current_log_pis,
                    cql_next_log_pis,
                ],
                dim=1,
            )
            # HACK: check dim
            cql_q_samples = cql_q_samples - importance_prob.unsqueeze(0)
        else:
            cql_q_samples = torch.cat([cql_q_samples, q_pred.unsqueeze(-1)], dim=-1)

            cql_q_samples -= (
                torch.log(
                    torch.tensor(
                        cql_q_samples.shape[-1],
                        dtype=torch.float,
                        device=cql_q_samples.device,
                    )
                )
                * self.config.cql_temp
            )
            # HACK: shape diff from jax implementation
            assert cql_q_samples.shape == (
                self.config.critic_ensemble_size,
                batch_size,
                3 * self.config.cql_n_actions + 1,
            )

        """log sum exp of the ood actions"""
        cql_ood_values = (
            torch.logsumexp(cql_q_samples / self.config.cql_temp, dim=-1)
            * self.config.cql_temp
        )
        assert cql_ood_values.shape == (self.config.critic_ensemble_size, batch_size)

        cql_q_diff = cql_ood_values - q_pred
        info = {
            "cql_ood_values": cql_ood_values.mean(),
        }
        if self.config.use_calql:
            info["calql_bound_rate"] = calql_bound_rate

        return cql_q_diff, info

    @staticmethod
    def make_optimizer(
        params: torch.nn.Module,
        learning_rate: float = 3e-4,
        warmup_steps: int = 0,
        cosine_decay_steps: Optional[int] = None,
        weight_decay: Optional[float] = None,
        return_lr_schedule: bool = True,
    ) -> Union[Optimizer, Tuple[Optimizer, LambdaLR]]:
        optimizer: Optimizer
        if weight_decay is not None:
            optimizer = AdamW(
                params=params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            optimizer = Adam(params=params, lr=learning_rate)

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps

            if cosine_decay_steps is not None:
                decay_step = step - warmup_steps
                if decay_step < 0:
                    return 0.0
                if decay_step >= cosine_decay_steps:
                    return 0.0
                progress = decay_step / cosine_decay_steps
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

        if return_lr_schedule:
            return optimizer, scheduler
        else:
            return optimizer

    def prepare_optimizers(self):
        actor_optimizer, actor_scheduler = self.make_optimizer(
            self.policy.parameters(), **self.config.actor_optimizer_kwargs
        )
        critic_optimizer, critic_scheduler = self.make_optimizer(
            self.critics.parameters(), **self.config.critic_optimizer_kwargs
        )
        temperature_optimizer, temperature_scheduler = self.make_optimizer(
            self.temperature.parameters(), **self.config.temperature_optimizer_kwargs
        )

        return (
            actor_optimizer,
            actor_scheduler,
            critic_optimizer,
            critic_scheduler,
            temperature_optimizer,
            temperature_scheduler,
        )

    def forward(self, batch: at.CalQlBatch):
        temperature_loss, temperature_loss_info = self.temperature_loss_fn(batch)
        policy_loss, policy_loss_info = self.policy_loss_fn(batch)
        critic_loss, critic_loss_info = self.critic_loss_fn(batch)

        return (
            temperature_loss,
            policy_loss,
            critic_loss,
            {
                **temperature_loss_info,
                **policy_loss_info,
                **critic_loss_info,
            },
        )

    @jaxtyped(typechecker=typechecker)
    def get_q_values(
        self,
        encoded_observations: Float[
            torch.Tensor, "batch {self.config.obs_encoded_dim}"
        ],
        noise_actions: Float[torch.Tensor, "batch num_actions action_dim"],
    ) -> Float[torch.Tensor, "batch num_actions"]:
        # (num_backbones, batch, *num_actions)
        q_values = self.target_critics.forward(encoded_observations, noise_actions)
        q_values = q_values.min(dim=0)[0]
        return q_values
