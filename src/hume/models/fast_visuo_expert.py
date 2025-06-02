from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    Dinov2Model,
    GemmaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING


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


class FastVisuoExpertConfig(PretrainedConfig):
    model_type = "FastVisuoExpertModel"
    sub_configs = {"dino_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        dino_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.attention_implementation = attention_implementation

        if dino_config is None:
            self.dino_config = CONFIG_MAPPING["dinov2"](
                transformers_version="4.48.1",
                model_type="dinov2",
                attention_probs_dropout_prob=0.0,
                drop_path_rate=0.0,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                hidden_size=384,
                image_size=518,
                initializer_range=0.02,
                layer_norm_eps=1e-06,
                layerscale_value=1.0,
                mlp_ratio=4,
                num_attention_heads=6,
                num_channels=3,
                num_hidden_layers=12,
                patch_size=14,
                qkv_bias=True,
                torch_dtype="float32",
                use_swiglu_ffn=False,
            )
        elif isinstance(dino_config, dict):
            if "model_type" not in dino_config:
                dino_config["model_type"] = "dinov2"
            cfg_cls = CONFIG_MAPPING[dino_config["model_type"]]
            self.dino_config = cfg_cls(**dino_config)

        if gemma_expert_config is None:
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=8,
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


class FastVisuoExpertModel(PreTrainedModel):
    config_class = FastVisuoExpertConfig

    def __init__(self, config: FastVisuoExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.vision_tower = Dinov2Model(config=config.dino_config)
        self.gemma_expert = GemmaForCausalLM(
            config=config.gemma_expert_config
        )  # GemmaModel
        self.multi_modal_projector = nn.Linear(
            config.dino_config.hidden_size, config.gemma_expert_config.hidden_size
        )
        self.gemma_expert.model.embed_tokens = None
        self.gemma_expert.lm_head = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.vision_tower.eval()
            for params in self.vision_tower.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.vision_tower.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.vision_tower = self.vision_tower.to(dtype=torch.bfloat16)
        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        selected_image_feature = self.vision_tower(image).last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (
            self.config.gemma_expert_config.hidden_size**0.5
        )
        return image_features

    # TODO: break down this huge forward into modules or functions
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
