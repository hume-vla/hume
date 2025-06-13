from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("hume")
@dataclass
class HumeConfig(PreTrainedConfig):
    # Input / output structure.
    type: str = "hume"
    n_obs_steps: int = 1
    s1_chunk_size: int = 10
    s2_chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # + Aadditional attributes for s1 / s2
    # freeze system
    freeze_s2: bool = False
    s1_his_state_size: int = 1
    cache_s2_actions: bool = False

    # denoise ratio
    theta2: float = 1.0
    theta1: float = 1.0
    noise_slides_eps: float = 0.0
    noise_slides_alp: float = 0.0

    # projector
    s1_proj_width: int = 512  # NOTE: consitent with the s1_gemma_expert_config
    freeze_s1_vision_encoder: bool = False

    # decoding
    s1_num_steps: int = 10

    # vqh
    num_pos: int = 3
    discount: float = 0.98
    actor_lr: float = 1e-4  # actor learning rate
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    qf_lr: float = 3e-4  # Critics learning rate
    next_obs_offset: int = 10  # should be equal to vqh_chunk_size
    vqh_chunk_size: int = 10

    paligemma_config: dict = field(
        default_factory=lambda: {
            "bos_token_id": 2,
            "eos_token_id": 1,
            "hidden_size": 2048,
            "ignore_index": -100,
            "image_token_index": 257152,
            "model_type": "paligemma",
            "pad_token_id": 0,
            "projection_dim": 2048,
            "text_config": {
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2048,
                "intermediate_size": 16384,
                "model_type": "gemma",
                "num_attention_heads": 8,
                "num_hidden_layers": 18,
                "num_image_tokens": 256,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
                "vocab_size": 257152,
            },
            "torch_dtype": "float32",
            "transformers_version": "4.48.1",
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "model_type": "siglip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "num_image_tokens": 256,
                "patch_size": 14,
                "projection_dim": 2048,
                "projector_hidden_act": "gelu_fast",
                "vision_use_head": False,
            },
            "vocab_size": 257152,
        }
    )

    gemma_expert_config: dict = field(
        default_factory=lambda: {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
            "transformers_version": "4.48.1",
            "use_cache": True,
            "vocab_size": 257152,
        }
    )

    # TODO: Add EMA

    # system2 configurations
    s1_dino_config: dict = field(
        default_factory=lambda: {
            "model_type": "dinov2",
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 384,
            "image_size": 518,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-06,
            "layerscale_value": 1.0,
            "mlp_ratio": 4,
            "num_attention_heads": 6,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "patch_size": 14,
            "qkv_bias": True,
            "torch_dtype": "float32",
            "use_swiglu_ffn": False,
        }
    )

    s1_gemma_expert_config: dict = field(
        default_factory=lambda: {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 13,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
            "transformers_version": "4.48.1",
            "use_cache": True,
            "vocab_size": 257152,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.s2_chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.s2_chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> dict[AdamWConfig]:
        qf_optimizer = AdamWConfig(
            lr=self.qf_lr,
            weight_decay=0,
            grad_clip_norm=10,
        )
        actor_optimizer = AdamWConfig(
            lr=self.actor_lr,
            weight_decay=0,
            grad_clip_norm=10,
        )

        trunk_optimizer = AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

        optimizer_dict = dict(
            qf_optimizer=qf_optimizer,
            actor_optimizer=actor_optimizer,
            trunk_optimizer=trunk_optimizer,
        )

        return optimizer_dict

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.s2_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def slide(self) -> None:
        return self.s2_chunk_size // self.s1_chunk_size

    @property
    def s1_action_steps(self) -> None:
        return self.s1_chunk_size

    @property
    def s2_action_steps(self) -> None:
        return self.s2_chunk_size


@PreTrainedConfig.register_subclass("system2")
@dataclass
class System2Config(PreTrainedConfig):
    # Input / output structure.
    num_pos: int = 3
    discount: float = 0.98
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    next_obs_offset: int = 1
    s1_his_state_size: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    paligemma_config: dict = field(
        default_factory=lambda: {
            "bos_token_id": 2,
            "eos_token_id": 1,
            "hidden_size": 2048,
            "ignore_index": -100,
            "image_token_index": 257152,
            "model_type": "paligemma",
            "pad_token_id": 0,
            "projection_dim": 2048,
            "text_config": {
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2048,
                "intermediate_size": 16384,
                "model_type": "gemma",
                "num_attention_heads": 8,
                "num_hidden_layers": 18,
                "num_image_tokens": 256,
                "num_key_value_heads": 1,
                "torch_dtype": "float32",
                "vocab_size": 257152,
            },
            "torch_dtype": "float32",
            "transformers_version": "4.48.1",
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "model_type": "siglip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "num_image_tokens": 256,
                "patch_size": 14,
                "projection_dim": 2048,
                "projector_hidden_act": "gelu_fast",
                "vision_use_head": False,
            },
            "vocab_size": 257152,
        }
    )

    gemma_expert_config: dict = field(
        default_factory=lambda: {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 8192,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "torch_dtype": "float32",
            "transformers_version": "4.48.1",
            "use_cache": True,
            "vocab_size": 257152,
        }
    )

    # TODO: Add EMA

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def slide(self) -> None:
        return 1

    @property
    def s1_action_steps(self) -> None:
        return 1
