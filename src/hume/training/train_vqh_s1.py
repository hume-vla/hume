import logging
import shutil
import time
from contextlib import nullcontext

# custom imports
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any
import datetime as dt
from pathlib import Path
import copy
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.default import DatasetConfig as LeroBotDatasetConfig
from lerobot.configs.train import TrainPipelineConfig as LeroBotTrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from termcolor import colored
from torch.optim import Optimizer
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hume.models.modeling_hume import HumePolicy, System2Policy
from hume.training.lerobot_patch import (
    make_dataset,
    make_optimizer_and_scheduler,
    make_policy,
    replace_save_optimizer_state,
    replace_save_scheduler_state,
)
from hume.training.transforms import ImageTransformsConfig

replace_save_optimizer_state()
replace_save_scheduler_state()


@dataclass
class DatasetConfig(LeroBotDatasetConfig):  # noqa: F811
    """Configuration for the dataset"""

    wrist_transforms: ImageTransformsConfig = field(
        default_factory=ImageTransformsConfig
    )


@dataclass
class TrainPipelineConfig(LeroBotTrainPipelineConfig):
    """Configuration for the training pipeline"""

    dataset: DatasetConfig

    policy_optimizer_lr: float = None
    actor_lr: float = 1e-4  # actor learning rate
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    checkpoints_total_limit: int = 2
    output_base: str = "hume"
    pretrained_s2_path: str = None
    pretrained_paligemma_path: str = None
    pretrained_dino_path: str = None
    pretrained_tokenizer_path: str = None
    freeze_vision_encoder: bool = True
    freeze_s2: bool = True
    empty_cameras: int = 0

    # hume
    theta1: float = 1.0
    theta2: float = 1.0
    s1_chunk_size: int = 8
    s2_chunk_size: int = 8
    noise_slides_eps: float = 0.0
    noise_slides_alp: float = 0.0

    # value_query_head
    vqh_chunk_size: int = 10
    num_pos: int = 3
    discount: float = 0.98
    soft_target_critic_update_rate: float = 5e-3  # Target network update rate
    target_critic_update_period: int = 1  # Frequency of target nets updates
    next_obs_offset: int = 10
    s1_his_state_size: int = 1
    cache_s2_actions: bool = False

    # # wandb
    # wandb: WandBConfig = field(default_factory=WandBConfig(entity="qudelin-org"))

    def validate(self):
        over_cfg = copy.deepcopy(self.policy)
        # NOTE: policy overwrite order: default > config_path >> path > cli
        super().validate()
        logging.info(colored("overwrite additional policy features", "yellow"))
        if over_cfg:
            self.policy.input_features, self.policy.output_features = (
                over_cfg.input_features,
                over_cfg.output_features,
            )

        if self.policy_optimizer_lr:
            self.policy.optimizer_lr = self.policy_optimizer_lr
        if self.actor_lr:
            self.policy.actor_lr = self.actor_lr
        if self.critic_lr:
            self.policy.critic_lr = self.critic_lr
        if self.temp_lr:
            self.policy.temp_lr = self.temp_lr
        self.policy.scheduler_decay_steps = self.steps
        self.policy.freeze_vision_encoder = self.freeze_vision_encoder
        self.policy.freeze_s2 = self.freeze_s2
        self.policy.theta1 = self.theta1
        self.policy.theta2 = self.theta2
        self.policy.noise_slides_eps = self.noise_slides_eps
        self.policy.noise_slides_alp = self.noise_slides_alp
        self.policy.s1_chunk_size = self.s1_chunk_size
        self.policy.s2_chunk_size = self.s2_chunk_size

        self.policy.n_action_steps = self.s2_chunk_size
        self.policy.vqh_chunk_size = self.vqh_chunk_size
        self.policy.empty_cameras = self.empty_cameras
        self.policy.num_pos = self.num_pos
        self.policy.discount = self.discount
        self.policy.next_obs_offset = self.next_obs_offset
        self.policy.s1_his_state_size = self.s1_his_state_size
        self.policy.cache_s2_actions = self.cache_s2_actions

        if not self.resume:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path(f"outputs/{self.output_base}") / train_dir

        if self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizers: dict[str, Optimizer],
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_schedulers=None,
    lock=None,
    cfg: TrainPipelineConfig = None,
) -> tuple[MetricsTracker, dict]:
    trunk_optimizer = optimizers["trunk_optimizer"]
    actor_optimizer = optimizers["actor_optimizer"]
    critic_optimizer = optimizers["critic_optimizer"]
    temperature_optimizer = optimizers["temperature_optimizer"]

    def update_target_network(policy, soft_target_update_rate: float):
        soft_update(
            policy.value_query_head.calql.target_critics,
            policy.value_query_head.calql.critics,
            soft_target_update_rate,
        )

    def soft_update(target: nn.Module, source: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )

    start_time = time.perf_counter()
    policy.train()
    with accelerator.accumulate(policy):
        chunk_loss, temperature_loss, policy_loss, critic_loss, output_dict = (
            policy.forward(batch)
        )

        # NOTE: backward together
        accelerator.backward(chunk_loss + temperature_loss + policy_loss + critic_loss)

        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(
                accelerator.unwrap_model(
                    policy, keep_fp32_wrapper=True
                ).get_trunk_params(),
                grad_clip_norm,
            )

        with lock if lock is not None else nullcontext():
            trunk_optimizer.step()
            actor_optimizer.step()
            critic_optimizer.step()
            temperature_optimizer.step()

        trunk_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        temperature_optimizer.zero_grad()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_schedulers is not None:
            lr_schedulers["trunk_scheduler"].step()
            lr_schedulers["actor_scheduler"].step()
            lr_schedulers["critic_scheduler"].step()
            lr_schedulers["temperature_scheduler"].step()

    if train_metrics.steps % cfg.target_critic_update_period == 0:
        update_target_network(
            policy=accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
            soft_target_update_rate=cfg.soft_target_critic_update_rate,
        )

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = chunk_loss.item()  # TODO: gather loss from all processes
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = trunk_optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    # initialize accelerator and logging
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    n = len(str(accelerator.num_processes))
    process_id = colored(f"[{accelerator.process_index:0{n}d}] ", "green")
    logging.info(
        colored(
            f"Enable acccelerate 🤗, Processes: [{accelerator.process_index}/{accelerator.num_processes}], Precision: [{accelerator.mixed_precision}], Accumulation [{accelerator.gradient_accumulation_steps}]",
            "yellow",
            attrs=["bold"],
        )
    )

    # disable logging on non-main processes.
    if not accelerator.is_main_process:
        cfg.wandb.enable = False

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(
            process_id
            + colored("Logs will be saved locally.", "yellow", attrs=["bold"])
        )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # check device is available
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg.policy.device = "cpu"

    logging.info(process_id + "Creating dataset")
    # TODO: support distributed dataset
    dataset = make_dataset(cfg)
    keys = list(
        filter(lambda x: x not in cfg.policy.input_features, dataset.meta.video_keys)
    )
    logging.info(colored(f"remove unused video features {keys}", "yellow"))
    for key in keys:
        dataset.features.pop(key)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and accelerator.is_main_process:
        logging.info(process_id + "Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info(process_id + "Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        policy_cls=HumePolicy,
    )

    if cfg.pretrained_s2_path:
        logging.info(process_id + "Loading pretrained s2 ...")

        system2 = System2Policy.from_pretrained(
            cfg.pretrained_s2_path, local_files_only=True
        )

        policy.s2_model.load_state_dict(system2.model.state_dict())
        policy.s2_model.to(device)
        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_s2_path
        )

    if cfg.pretrained_paligemma_path:
        logging.info(process_id + "Loading pretrained paligemma ...")
        policy.config.paligemma_config = AutoConfig.from_pretrained(
            cfg.pretrained_paligemma_path
        ).to_dict()
        policy.s2_model.paligemma_with_expert.paligemma = AutoModel.from_pretrained(
            cfg.pretrained_paligemma_path, device_map=device
        )
        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_paligemma_path
        )
        # post init
        policy.s2_model.paligemma_with_expert.to_bfloat16_like_physical_intelligence()
        policy.s2_model.paligemma_with_expert.set_requires_grad()

    if cfg.pretrained_dino_path:
        logging.info(process_id + "Loading pretrained dino ...")
        policy.config.s1_dino_config = AutoConfig.from_pretrained(
            cfg.pretrained_dino_path
        ).to_dict()
        policy.s1_model.fast_visuo_expert.vision_tower = AutoModel.from_pretrained(
            cfg.pretrained_dino_path, device_map=device
        )
        policy.s1_model.fast_visuo_expert.to_bfloat16_like_physical_intelligence()
        policy.s1_model.fast_visuo_expert.set_requires_grad()

    if policy.language_tokenizer is None:
        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_tokenizer_path
        )

    logging.info(process_id + "Creating optimizer and scheduler")
    optimizers, lr_schedulers = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizers, lr_schedulers = load_training_state(
            cfg.checkpoint_path, optimizers, lr_schedulers
        )

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())
    num_s1_params = sum(p.numel() for p in policy.s1_model.parameters())
    num_s2_params = sum(p.numel() for p in policy.s2_model.parameters())

    logging.info(
        process_id
        + colored("Output dir:", "yellow", attrs=["bold"])
        + f" {cfg.output_dir}"
    )
    if cfg.env is not None:
        logging.info(process_id + f"{cfg.env.task=}")
    logging.info(process_id + f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(
        process_id + f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})"
    )
    logging.info(process_id + f"{dataset.num_episodes=}")
    logging.info(process_id + f"{num_s1_params=} ({format_big_number(num_s1_params)})")
    logging.info(process_id + f"{num_s2_params=} ({format_big_number(num_s2_params)})")
    logging.info(
        process_id
        + f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
    )
    logging.info(
        process_id + f"{num_total_params=} ({format_big_number(num_total_params)})"
    )

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    (
        policy,
        trunk_optimizer,
        actor_optimizer,
        critic_optimizer,
        temperature_optimizer,
        dataloader,
        trunk_scheduler,
        actor_scheduler,
        critic_scheduler,
        temperature_scheduler,
    ) = accelerator.prepare(
        policy, *tuple(optimizers.values()), dataloader, *tuple(lr_schedulers.values())
    )

    optimizers = dict(
        trunk_optimizer=trunk_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        temperature_optimizer=temperature_optimizer,
    )
    lr_schedulers = dict(
        trunk_scheduler=trunk_scheduler,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        temperature_scheduler=temperature_scheduler,
    )

    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    if accelerator.is_main_process:
        logging.info(colored(pformat(cfg.to_dict()), "blue", attrs=["bold"]))
        # cfg.save_pretrained(cfg.output_dir)

    logging.info(process_id + "Start offline training on a fixed dataset")

    progress_bar = tqdm(
        range(0, cfg.steps // accelerator.num_processes),
        initial=0,
        desc="steps",
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(step, cfg.steps // accelerator.num_processes):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizers,
            cfg.optimizer["trunk_optimizer"].grad_clip_norm,
            accelerator=accelerator,
            lr_schedulers=lr_schedulers,
            cfg=cfg,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        progress_bar.update(1)
        progress_bar.set_postfix(output_dict, epoch=train_tracker.epochs)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = (
            step % cfg.save_freq == 0 or step == cfg.steps // accelerator.num_processes
        )
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(process_id + str(train_tracker))
            if wandb_logger:
                wandb_log_dict = {**train_tracker.to_dict(), **output_dict}
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step and accelerator.is_main_process:
            logging.info(process_id + f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir,
                step,
                cfg,
                accelerator.unwrap_model(policy),
                optimizers,
                lr_schedulers,
            )

            # update_last_checkpoint(checkpoint_dir) # BUG: simlink not supported in oss
            if cfg.checkpoints_total_limit > 0:
                checkpoint_paths = sorted(checkpoint_dir.parent.iterdir())
                for checkpoint_path in checkpoint_paths[: -cfg.checkpoints_total_limit]:
                    try:
                        shutil.rmtree(checkpoint_path)
                    except:  # noqa: E722
                        print(f"failed to remove {checkpoint_path}")

            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(process_id + f"Eval policy at step {step}")
                with torch.no_grad():
                    eval_info = eval_policy(
                        eval_env,
                        accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    num_step=cfg.steps,
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop(
                    "avg_sum_reward"
                )
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(process_id + eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(
                        eval_info["video_paths"][0], step, mode="eval"
                    )
            accelerator.wait_for_everyone()

    if eval_env:
        eval_env.close()
    logging.info(process_id + "End of training")
    # Online training.
    # TODO: @Online training.
    accelerator.end_training()


if __name__ == "__main__":
    init_logging()
    train()
