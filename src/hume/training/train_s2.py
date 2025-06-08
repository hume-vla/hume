import copy  # noqa: E402
import datetime as dt
import logging
import shutil
import time
from contextlib import nullcontext

# custom imports
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
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
from lerobot.configs.default import DatasetConfig as LeroBotDatasetConfig
from lerobot.configs.train import TrainPipelineConfig as LeroBotTrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from termcolor import colored
from torch.optim import Optimizer
from tqdm.auto import tqdm

from hume.models.modeling_hume import System2Policy
from hume.training.lerobot_patch import make_dataset, make_policy
from hume.training.transforms import ImageTransformsConfig


@dataclass
class DatasetConfig(LeroBotDatasetConfig):
    """Configuration for the dataset"""

    wrist_transforms: ImageTransformsConfig = field(
        default_factory=ImageTransformsConfig
    )


@dataclass
class TrainPipelineConfig(LeroBotTrainPipelineConfig):
    """Configuration for the training pipeline"""

    dataset: DatasetConfig
    pretrained_tokenizer_path: str = None
    policy_optimizer_lr: float = None
    checkpoints_total_limit: int = 2
    chunk_size: int = 50
    output_base: str = "hume_s2"
    pretrained_paligemma_path: str = None
    freeze_vision_encoder: bool = True
    attention_implementation: str = "eager"
    empty_cameras: int = 0

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
        self.policy.scheduler_decay_steps = self.steps
        self.policy.freeze_vision_encoder = self.freeze_vision_encoder

        self.policy.n_action_steps = self.policy.chunk_size = self.chunk_size
        self.policy.attention_implementation = self.attention_implementation
        self.policy.empty_cameras = self.empty_cameras
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
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    with accelerator.accumulate(policy):
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()  # TODO: gather loss from all processes
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
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
        policy_cls=System2Policy,
    )

    if cfg.pretrained_paligemma_path:
        from transformers import (
            AutoTokenizer,
            PaliGemmaConfig,
            PaliGemmaForConditionalGeneration,
        )

        logging.info(process_id + "Loading customized models ...")
        paligemma_config = PaliGemmaConfig.from_pretrained(
            cfg.pretrained_paligemma_path
        )
        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.pretrained_paligemma_path, device_map=device
        )

        policy.model.paligemma_with_expert.paligemma = paligemma
        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_paligemma_path
        )

        policy.model.paligemma_with_expert.to_bfloat16_like_physical_intelligence()
        policy.model.paligemma_with_expert.set_requires_grad()
        policy.config.paligemma_config = paligemma_config.to_dict()
    elif not cfg.policy.pretrained_path:
        from transformers import AutoTokenizer

        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_tokenizer_path
        )

    logging.info(process_id + "Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

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

    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
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
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
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
                optimizer,
                lr_scheduler,
            )
            # update_last_checkpoint(checkpoint_dir) # BUG: simlink not supported in oss
            if cfg.checkpoints_total_limit > 0:
                checkpoint_paths = sorted(checkpoint_dir.parent.iterdir())
                for checkpoint_path in checkpoint_paths[: -cfg.checkpoints_total_limit]:
                    try:
                        shutil.rmtree(checkpoint_path)
                    except Exception:
                        print(f"failed to remove {checkpoint_path}")  # noqa: E722

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
