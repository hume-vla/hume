import logging
from pathlib import Path
from pprint import pformat

import lerobot
import lerobot.common
import lerobot.common.utils
import lerobot.common.utils.train_utils
import torch
from lerobot.common.constants import OPTIMIZER_PARAM_GROUPS, SCHEDULER_STATE
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    dataset_to_policy_features,
    flatten_dict,
    write_json,
)
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import env_to_policy_features

# from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from safetensors.torch import save_file
from termcolor import colored
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hume.training.dataset import LeRobotDataset
from hume.training.transforms import ImageTransforms

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    **kwargs,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError(
            "Either one of a dataset metadata or a sim env must be provided."
        )

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = kwargs.pop("policy_cls")

    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    # NOTE: do not override the input/output features if they are already set in the config.
    if not cfg.output_features:
        cfg.output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    print(
        colored(
            f"input_features: {cfg.input_features}, \noutput_features: {cfg.output_features} 🤗",
            "yellow",
            attrs=["bold"],
        )
    )

    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy


def make_dataset(
    cfg: TrainPipelineConfig, **kwargs
) -> LeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )
    wrist_transforms = (
        ImageTransforms(cfg.dataset.wrist_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            wrist_transforms=wrist_transforms,
            video_backend=cfg.dataset.video_backend,
            slide=cfg.policy.slide,
            s1_action_steps=cfg.policy.s1_action_steps,
            num_pos=cfg.policy.num_pos,
            discount=cfg.policy.discount,
            next_obs_offset=cfg.policy.next_obs_offset,
            s1_his_state_size=cfg.policy.s1_his_state_size,
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    return dataset


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[list[Optimizer], LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = (
        policy.get_trunk_params()
        if cfg.use_policy_training_preset
        else policy.parameters()
    )

    trunk_optimizer = cfg.optimizer["trunk_optimizer"].build(params)
    (
        actor_optimizer,
        actor_scheduler,
        critic_optimizer,
        critic_scheduler,
        temperature_optimizer,
        temperature_scheduler,
    ) = policy.value_query_head.calql.prepare_optimizers()

    lr_schedulers = dict(
        trunk_scheduler=cfg.scheduler.build(trunk_optimizer, cfg.steps)
        if cfg.scheduler is not None
        else None,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        temperature_scheduler=temperature_scheduler,
    )

    optimizers = dict(
        trunk_optimizer=trunk_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        temperature_optimizer=temperature_optimizer,
    )

    return optimizers, lr_schedulers


# HACK: replace lerobot save_optimizer_state as monkey patch
def save_optimizer_state(
    optimizers: dict[str, torch.optim.Optimizer], save_dir: Path
) -> None:
    param_groups = {}
    for optimizer_name, optimizer in optimizers.items():
        state = optimizer.state_dict()
        param_groups[optimizer_name] = state.pop("param_groups")
        flat_state = flatten_dict(state)
        save_file(flat_state, save_dir / f"{optimizer_name}_state.safetensors")
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def replace_save_optimizer_state():
    lerobot.common.utils.train_utils.save_optimizer_state = save_optimizer_state
    print("Repalce save_optimizer_state")


def save_scheduler_state(scheduler: dict[str, LRScheduler], save_dir: Path) -> None:
    multi_scheduler_state_dict = {}
    for scheduler_name, _scheduler in scheduler.items():
        multi_scheduler_state_dict[scheduler_name] = _scheduler.state_dict()

    write_json(multi_scheduler_state_dict, save_dir / SCHEDULER_STATE)


def replace_save_scheduler_state():
    lerobot.common.utils.train_utils.save_scheduler_state = save_scheduler_state
    print("Repalce save_optimizer_state")


# TODO: Add deserialize method for load multi optimizer state
