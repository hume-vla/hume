import collections
import math
from argparse import Namespace
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int64, jaxtyped
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype
from torch import Tensor, nn
from transformers import AutoTokenizer

from .. import array_typing as at
from .configuration_hume import HumeConfig, System2Config
from .fast_visuo_expert import FastVisuoExpertConfig, FastVisuoExpertModel
from .paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from .value_query import (
    CalQL,
    CalQlConfig,
    VQHBackbone,
    VQHBackboneConfig,
)


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class HumePolicy(PreTrainedPolicy):
    """Wrapper class around System2 model to train and run inference within LeRobot."""

    config_class = HumeConfig
    name = "hume"

    def __init__(
        self,
        config: HumeConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # TODO: input / output features / normalizer for mutiple datasets
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = None
        self.s2_model = System2(config)
        self.s1_model = FastVisuoMatching(config)
        self.value_query_head = ValueQueryHead(
            paligemma_with_expert=self.s2_model.paligemma_with_expert, config=config
        )
        self.reset()

        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_s2:
            self.s2_model.eval()
            for params in self.s2_model.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_s2:
            self.s2_model.eval()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.s2_action_cache = {}

    def get_trunk_params(self) -> dict:
        exclude_params = set()
        exclude_modules = [
            self.value_query_head.calql.policy,
            self.value_query_head.calql.critics,
            self.value_query_head.calql.temperature,
        ]

        for module in exclude_modules:
            for param in module.parameters():
                exclude_params.add(id(param))

        return [param for param in self.parameters() if id(param) not in exclude_params]

    def get_optim_params(self) -> dict:
        return self.parameters()

    def get_actor_optim_params(self) -> dict:
        return self.value_query_head.calql.policy.parameters()

    def get_critics_optim_params(self) -> dict:
        return self.value_query_head.calql.critics.parameters()

    def get_temperature_optim_params(self) -> dict:
        return self.value_query_head.calql.temperature.parameters()

    def init_infer(self, infer_cfg: at.InferConfig):
        self.infer_cfg = Namespace(**infer_cfg)
        self.action_plan = collections.deque()
        self.history_state = collections.deque(maxlen=self.config.s1_his_state_size)
        self.infer_step = 0
        self.outputs = {}
        self.q_value_cache = []
        self.action_cache = []

        self.reset()
        print("Initializing inference with config:", infer_cfg)

        return True

    def infer(self, observation: at.InferBatchObs) -> at.ActionArray:
        # prcoess observation
        # from np.array -> torch.tensor -> add batch, change shape
        if not self.history_state:
            self.history_state.extend(
                np.expand_dims(observation["observation.state"], 1)
                .repeat(self.config.s1_his_state_size, axis=1)
                .transpose(1, 0, 2)
            )
        else:
            self.history_state.append(observation["observation.state"])

        observation["observation.state"] = np.asarray(self.history_state).transpose(
            1, 0, 2
        )

        observation: dict[str, torch.tensor | list[str]] = {
            **{
                k: torch.tensor(v / 255)  # b, h, w ,c
                .permute(0, 3, 1, 2)  # b, c, h, w
                .to(self.infer_cfg.device)
                .float()
                for k, v in observation.items()
                if k
                in {
                    "observation.images.image",
                    "observation.images.wrist_image",
                    "observation.images.image_0",
                }
            },
            **{k: v for k, v in observation.items() if k in {"task"}},  # len = batch
            **{
                k: torch.tensor(v)
                .to(self.infer_cfg.device)
                .float()  # b, state_horizon, state_dim
                for k, v in observation.items()
                if k in {"observation.state"}
            },
        }
        batch_size = len(observation["task"])

        if not self.action_plan:
            # Finished executing previous action chunk -- compute new chunk
            # Prepare observations dict
            # infer the action
            if self.infer_step % self.infer_cfg.s2_replan_steps == 0:
                self.outputs = {}  # infer with s1 or s2
            stamp = (
                torch.tensor(
                    [
                        self.infer_step
                        % self.infer_cfg.s2_replan_steps
                        / self.config.s2_chunk_size
                    ]
                )
                .expand(batch_size)
                .to(self.infer_cfg.device)
                .float()
            )
            self.outputs = self.select_action(
                observation,
                self.outputs,
                stamp,
                s2_candidates_num=self.infer_cfg.s2_candidates_num,
                noise_temp_bounds=(
                    self.infer_cfg.noise_temp_lower_bound,
                    self.infer_cfg.noise_temp_upper_bound,
                ),
                time_temp_bounds=(
                    self.infer_cfg.time_temp_lower_bound,
                    self.infer_cfg.time_temp_upper_bound,
                ),
            )
            action_chunk = self.outputs["s1_action"].cpu().numpy()

            if self.infer_cfg.post_process_action:
                action_chunk[..., -1] = 2 * (1 - action_chunk[..., -1]) - 1

            # convert action chunk shape to (replan_steps, batch, action_dim)
            action_chunk = action_chunk.transpose(1, 0, 2)
            assert len(action_chunk) >= self.infer_cfg.replan_steps, (
                f"We want to replan every {self.infer_cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            )
            self.action_plan.extend(action_chunk[: self.infer_cfg.replan_steps])

        self.infer_step += 1
        action = self.action_plan.popleft()
        return np.asarray(action)

    @torch.no_grad
    @jaxtyped(typechecker=typechecker)
    def select_action(
        self,
        batch: at.InferBatchObs,
        outputs: at.InferOutput = {},
        stamp: Float[Tensor, " batch"] | None = None,
        s2_candidates_num: int = 5,
        noise_temp_bounds: tuple = (1.0, 1.0),
        time_temp_bounds: tuple = (1.0, 1.0),
    ) -> at.InferOutput:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        batch = self.normalize_inputs(batch)

        # querying the policy.
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        original_action_dim = self.config.action_feature.shape[0]

        if "noise_action" not in outputs:
            noise_actions = []  # [(Batch, Chunksize, Action dim),]
            for i in range(s2_candidates_num):
                noise_actions.append(
                    self.s2_model.sample_actions(
                        images,
                        img_masks,
                        lang_tokens,
                        lang_masks,
                        state[:, -1, :],  # s2 not supported history state yet
                        time_temp=(i / s2_candidates_num)
                        * (time_temp_bounds[1] - time_temp_bounds[0])
                        + time_temp_bounds[0],
                        noise_temp=(i / s2_candidates_num)
                        * (noise_temp_bounds[1] - noise_temp_bounds[0])
                        + noise_temp_bounds[0],
                    )
                )
            noise_actions = torch.stack(noise_actions, dim=1)
            # (Batch, s2_candidates_num, Chunksize, Actiondim)
            batch_size = noise_actions.shape[0]
            batch_idx = torch.arange(batch_size, device=noise_actions.device)

            noise_actions_wo_pad = noise_actions[
                :, :, : self.config.vqh_chunk_size, :original_action_dim
            ]
            action_index, q_values = self.value_query_head.select_q_actions(
                images, img_masks, lang_tokens, lang_masks, noise_actions_wo_pad
            )
            self.q_value_cache.append(q_values.squeeze())
            unnormalized_noise_actions = self.unnormalize_outputs(
                {"action": noise_actions_wo_pad}
            )["action"]
            self.action_cache.append(unnormalized_noise_actions.squeeze())
            selected_noise_action = noise_actions[batch_idx, action_index]

            outputs = {"noise_action": selected_noise_action}

        noise_action: Float[Tensor, "batch s2_chunksize action_dim"] = outputs[
            "noise_action"
        ]
        idcs = (stamp * self.config.s2_chunk_size).long().unsqueeze(1) + torch.arange(
            self.config.s1_chunk_size, device=noise_action.device
        )
        batch_idcs = torch.arange(
            noise_action.shape[0], device=noise_action.device
        ).unsqueeze(1)
        noise_action_slides = noise_action[batch_idcs, idcs]
        s1_actions = self.s1_model.sample_actions(
            images, img_masks, state, noise_action_slides, stamp=stamp
        )

        # Unpad actions
        actions = s1_actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({"action": actions})["action"]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions_inv(actions)

        outputs["s1_action"] = actions

        return outputs

    def post_normalize(self, batch):
        """additional keys {obervation.x}.s1 are merged in to the batch,
        so we need to normalize these keys
        """
        merge_keys = filter(lambda k: k.endswith(".s1"), batch.keys())
        for k in merge_keys:
            _k = k.replace(".s1", "")
            batch[k] = self.normalize_inputs({_k: batch[k]})[_k]
        return batch

    def get_noise_action_slides(self, action: Tensor, stamp: Tensor) -> Tensor:
        """Augment the action with the previous actions in the queue."""
        # idcs = (torch.rand_like(stamp) * (self.config.s2_chunk_size - self.config.s1_chunk_size)).long()
        idcs = (
            (
                self.config.noise_slides_alp * torch.rand_like(stamp)
                - self.config.noise_slides_alp / 2
                + stamp
            )
            * self.config.s2_chunk_size
        ).long()
        idcs = torch.clamp(idcs, 0, action.shape[1] - self.config.s1_chunk_size)
        idcs = idcs + torch.arange(self.config.s1_chunk_size, device=action.device)
        batch_idcs = torch.arange(action.shape[0], device=action.device).unsqueeze(1)
        noise_action_slides = action[batch_idcs, idcs]

        noise_action_slides += (
            torch.randn_like(noise_action_slides) * self.config.noise_slides_eps
        )
        return noise_action_slides

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.post_normalize(batch)
        batch = self.normalize_targets(batch)

        # prepare images
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        s1_images, s1_img_masks = self.prepare_images(
            batch, map(lambda x: f"{x}.s1", self.config.image_features)
        )  # 0
        s1_state = self.prepare_state(batch, f"{OBS_ROBOT}.s1")

        # prepare actions
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        b, s, _ = actions.shape
        device = actions.device
        batch_idcs = torch.arange(b, device=device).unsqueeze(1)
        stamp = batch["stamp"]
        idcs = (stamp * self.config.s2_chunk_size).long() + torch.arange(
            self.config.s1_chunk_size, device=device
        )
        s1_actions = actions[batch_idcs, idcs]
        s1_actions_is_pad = (
            None if actions_is_pad is None else actions_is_pad[batch_idcs, idcs]
        )

        # s2 forward pass
        with torch.no_grad():
            if self.config.cache_s2_actions:
                is_noised = []
                noise_actions = torch.zeros_like(actions)
                for idx, s2_idx in enumerate(batch["s2_idx"]):
                    if s2_idx in self.s2_action_cache:
                        noise_actions[idx] = self.s2_action_cache[s2_idx]
                        is_noised.append(False)
                    else:
                        is_noised.append(True)
                # noise batch
                is_noised = torch.tensor(is_noised, device=batch["s2_idx"].device)
                s2_actions_infered = self.s2_model.sample_actions(
                    [img[is_noised] for img in images],
                    [mask[is_noised] for mask in img_masks],
                    lang_tokens[is_noised],
                    lang_masks[is_noised],
                    state[is_noised],
                )
                noise_actions[is_noised] = s2_actions_infered
            else:
                noise_actions = self.s2_model.sample_actions(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                )

        # vgps: embs[q] -> layers -> [q] -> mlp
        # value query head features are end with vqh: xx.vqh
        vqh_images, vqh_img_masks = self.prepare_images(
            batch, map(lambda x: f"{x}.vqh", self.config.image_features)
        )  # 1

        temperature_loss, policy_loss, critic_loss, log_dict = (
            self.value_query_head.forward(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                vqh_images,
                vqh_img_masks,
                batch["action"][:, : self.config.vqh_chunk_size, :],
                batch["reward.vqh"],
                batch["mc.vqh"],
                batch["reward.vqh"].to(dtype=torch.float),
            )
        )

        noise_action_slides = self.get_noise_action_slides(noise_actions, stamp)
        s1_losses = self.s1_model.forward(
            s1_images,
            s1_img_masks,
            s1_state,
            s1_actions,
            noise_action_slides,
            time,
            stamp=stamp.squeeze(),
        )

        total_loss, loss_dict = 0.0, {}

        if s1_actions_is_pad is not None:
            in_episode_bound = ~s1_actions_is_pad
            s1_losses = s1_losses * in_episode_bound.unsqueeze(-1)

        s1_losses = s1_losses[..., : self.config.max_action_dim]
        s1_losses = s1_losses.mean()

        loss_dict["s1_loss"] = s1_losses.item()
        total_loss += s1_losses

        # add ValueQueryHead log dict to loss_dict
        # loss_dict = {**loss_dict, **log_dict}
        loss_dict["entropy"] = log_dict["entropy"].item()
        loss_dict["actions_mse"] = log_dict["actions_mse"].item()
        loss_dict["td_err"] = log_dict["td_err"].item()
        loss_dict["temperature"] = log_dict["temperature"].item()
        loss_dict["cql_loss"] = log_dict["cql_loss"].item()
        loss_dict["cql_alpha"] = log_dict["cql_alpha"]
        loss_dict["cql_diff"] = log_dict["cql_diff"].item()
        loss_dict["critic_loss"] = log_dict["critic_loss"].item()
        loss_dict["cql_ood_values"] = log_dict["cql_ood_values"].item()
        loss_dict["calql_bound_rate"] = log_dict["calql_bound_rate"].item()
        loss_dict["online_q"] = log_dict["online_q"].item()
        loss_dict["target_q"] = log_dict["target_q"].item()
        loss_dict["positive_qs"] = log_dict["positive_qs"].item()
        loss_dict["actor_loss"] = log_dict["actor_loss"].item()

        return total_loss, temperature_loss, policy_loss, critic_loss, loss_dict

    def prepare_images(self, batch, image_features=None):
        """Apply preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        image_features = image_features or self.config.image_features
        present_img_keys = [key for key in image_features if key in batch]
        missing_img_keys = [key for key in image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(
                    img, *self.config.resize_imgs_with_padding, pad_value=0
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(
            device=device, dtype=torch.bool
        )

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(
                actions[:, :, motor_idx]
            )
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(
                actions[:, :, motor_idx]
            )
        return actions

    def prepare_state(self, batch, feature=None):
        """Pad state"""
        feature = feature or OBS_ROBOT
        state = pad_vector(batch[feature], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _save_pretrained(self, save_directory) -> None:
        super()._save_pretrained(save_directory)
        print(f"Saving the language tokenizer to {save_directory} ...")
        self.language_tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        **kwargs,
    ):
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        print(f"Loading the language tokenizer from {pretrained_name_or_path} ...")
        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name_or_path
        )
        return policy


class System2Policy(PreTrainedPolicy):
    """Wrapper class around System2FlowMatching model to train and run inference within LeRobot."""

    config_class = System2Config
    name = "system2"

    def __init__(
        self,
        config: System2Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        # TODO: input / output features / normalizer for mutiple datasets
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = None
        self.model = System2(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"action": actions})["action"]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses, _ = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        # loss_dict["losses_after_forward"] = losses.detach().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            # loss_dict["losses_after_in_ep_bound"] = losses.detach().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        # loss_dict["losses_after_rm_padding"] = losses.detach().mean().item()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [
            key for key in self.config.image_features if key not in batch
        ]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(
                    img, *self.config.resize_imgs_with_padding, pad_value=0
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(
            device=device, dtype=torch.bool
        )

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(
                actions[:, :, motor_idx]
            )
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(
                actions[:, :, motor_idx]
            )
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _save_pretrained(self, save_directory) -> None:
        super()._save_pretrained(save_directory)
        print(f"Saving the language tokenizer to {save_directory} ...")
        self.language_tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        **kwargs,
    ):
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        print(f"Loading the language tokenizer from {pretrained_name_or_path} ...")
        policy.language_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name_or_path
        )
        return policy


class System2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            paligemma_config=self.config.paligemma_config,
            gemma_expert_config=self.config.gemma_expert_config,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_with_export_config
        )

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, self.config.proj_width
        )
        self.action_out_proj = nn.Linear(
            self.config.proj_width, self.config.max_action_dim
        )

        self.action_time_mlp_in = nn.Linear(
            self.config.proj_width * 2, self.config.proj_width
        )
        self.action_time_mlp_out = nn.Linear(
            self.config.proj_width, self.config.proj_width
        )

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=True,
            fill_kv_cache=True,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        return losses, past_key_values

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        past_key_values=None,
        time_temp=1.0,
        noise_temp=1.0,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        if past_key_values is None:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.config.use_cache,
                fill_kv_cache=True,
            )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(
            time_temp, dtype=torch.float32, device=device
        )  # TODO: Add temp
        while time >= -dt / 2 + (1 - self.config.theta2):
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t * noise_temp  # TODO: Add noise temp
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t


class FastVisuoMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # FastVisuoExpertConfig, FastVisuoExpertModel
        fast_visuo_expertConfig = FastVisuoExpertConfig(
            freeze_vision_encoder=self.config.freeze_s1_vision_encoder,
            attention_implementation=self.config.attention_implementation,
            dino_config=self.config.s1_dino_config,
            gemma_expert_config=self.config.s1_gemma_expert_config,
        )
        self.fast_visuo_expert = FastVisuoExpertModel(fast_visuo_expertConfig)

        # Projections are float32
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.config.s1_proj_width
        )
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, self.config.s1_proj_width
        )
        self.action_out_proj = nn.Linear(
            self.config.s1_proj_width, self.config.max_action_dim
        )
        self.action_time_mlp_in = nn.Linear(
            self.config.s1_proj_width * 2, self.config.s1_proj_width
        )
        self.action_time_mlp_out = nn.Linear(
            self.config.s1_proj_width, self.config.s1_proj_width
        )

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for img, img_mask in zip(images, img_masks, strict=False):
            DINO_MEAN, DINO_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            img = TF.normalize(img * 0.5 + 0.5, mean=DINO_MEAN, std=DINO_STD)
            img_emb = self.fast_visuo_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, stamp):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        state_horizon = state_emb.shape[1]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, state_horizon, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * state_horizon

        # Embed stamp
        stamp_emb = create_sinusoidal_pos_embedding(
            stamp,
            self.config.s1_proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        stamp_emb = stamp_emb.type(dtype=dtype)[:, None, :]
        embs.append(stamp_emb)
        stamp_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(stamp_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config.s1_proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.s1_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, state, actions, noise=None, time=None, stamp=None
    ) -> Float[
        Tensor, "batch {self.config.s1_action_steps} {self.config.max_action_dim}"
    ]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = (
                self.sample_time(actions.shape[0], actions.device) * self.config.theta1
            )
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time, stamp
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        inputs_embeds = torch.cat(
            [prefix_embs, suffix_embs], dim=1
        )  # torch.Size([16, 565]), torch.Size([16, 565])

        suffix_out = self.fast_visuo_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        suffix_out = suffix_out[:, -self.config.s1_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self, images, img_masks, state, noise=None, stamp=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.s1_action_steps,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        if stamp is None:
            stamp = torch.rand(bsize, device=device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )

        dt = -self.config.theta1 / self.config.s1_num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(self.config.theta1, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                expanded_time,
                stamp,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
        stamp,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep, stamp
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        inputs_embeds = torch.cat(
            [prefix_embs, suffix_embs], dim=1
        )  # torch.Size([16, 565]), torch.Size([16, 565])
        suffix_out = self.fast_visuo_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        suffix_out = suffix_out[:, -self.config.s1_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t


class ValueQueryHead(nn.Module):
    def __init__(self, paligemma_with_expert, config):
        super().__init__()
        # gemma_expert for processing img and languge tokens
        # paligemma with export fot processing image features
        self.config = config
        self.paligemma_with_expert = paligemma_with_expert

        vqh_backbone_config = VQHBackboneConfig()
        self.vqh_backbone = VQHBackbone(config=vqh_backbone_config)

        cal_ql_config = CalQlConfig(
            obs_encoded_dim=self.paligemma_with_expert.config.paligemma_config.hidden_size,
            action_dim=config.vqh_chunk_size * config.action_feature.shape[0],
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            temp_lr=config.temp_lr,
        )
        self.calql = CalQL(config=cal_ql_config)

        self.query_embedding = nn.Parameter(
            torch.zeros(
                self.paligemma_with_expert.config.paligemma_config.hidden_size,
                dtype=torch.bfloat16,
            )
        )

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(
            lang_tokens
        ).detach()

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # NOTE: add query embedding for each sequence
        seq_lengths = pad_masks.sum(dim=1).long()  # w/o padding length
        seq_len = embs.shape[1]

        new_seq_len = seq_len + 1
        new_embs = torch.zeros(
            (bsize, new_seq_len, embs.shape[-1]), dtype=embs.dtype, device=embs.device
        )
        new_pad_masks = torch.zeros(
            (bsize, new_seq_len), dtype=pad_masks.dtype, device=pad_masks.device
        )
        new_att_masks = torch.zeros(
            (bsize, new_seq_len), dtype=att_masks.dtype, device=att_masks.device
        )

        batch_idx = torch.arange(bsize, device=embs.device).view(-1, 1)
        seq_idx = (
            torch.arange(seq_len, device=embs.device).view(1, -1).expand(bsize, -1)
        )

        mask = seq_idx >= seq_lengths.unsqueeze(1)
        new_seq_idx = seq_idx + mask.long()

        new_embs[batch_idx, new_seq_idx] = embs
        new_pad_masks[batch_idx, new_seq_idx] = pad_masks
        new_att_masks[batch_idx, new_seq_idx] = att_masks
        new_embs[torch.arange(bsize), seq_lengths] = self.query_embedding.unsqueeze(
            0
        ).expand(bsize, -1)
        new_pad_masks[torch.arange(bsize), seq_lengths] = True
        new_att_masks[torch.arange(bsize), seq_lengths] = False

        return new_embs, new_pad_masks, new_att_masks

    def process_next_obs(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        vqh_images: list[torch.Tensor],
        vqh_img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Process next observation for ValueQueryHead model.
        Args:
            images (list): List of image tensors.
            img_masks (list): List of image mask tensors.
            vqh_images (list): List of ValueQueryHead image tensors.
            vqh_img_masks (list): List of ValueQueryHead image mask tensors.
            lang_tokens (torch.Tensor): Language token tensor.
            lang_masks (torch.Tensor): Language mask tensor.

        Returns:
            tuple: Tuple containing processed images, masks, and language tokens.
        """
        new_images = []
        new_img_masks = []

        for img, next_img, img_mask, next_img_mask in zip(
            images, vqh_images, img_masks, vqh_img_masks
        ):
            new_images.append(torch.cat([img, next_img], dim=0))
            new_img_masks.append(torch.cat([img_mask, next_img_mask], dim=0))

        new_lang_tokens = torch.cat([lang_tokens, lang_tokens], dim=0)
        new_lang_masks = torch.cat([lang_masks, lang_masks], dim=0)

        return (
            new_images,
            new_img_masks,
            new_lang_tokens,
            new_lang_masks,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        images: list[Float[Tensor, "batch 3 224 224"]],
        img_masks: list[Bool[Tensor, " batch"]],
        lang_tokens: Int64[Tensor, "batch seq_len"],
        lang_masks: Bool[Tensor, "batch seq_len"],
        vqh_images: list[Float[Tensor, "batch 3 224 224"]],
        vqh_img_masks: list[Bool[Tensor, " batch"]],
        actions: Float[
            Tensor,
            "batch {self.config.vqh_chunk_size} {self.config.action_feature.shape[0]}",
        ],
        rewards: Float[Tensor, " batch"],
        mc_returns: Float[Tensor, " batch"],
        masks: Float[Tensor, " batch"],
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Forward pass for ValueQueryHead model.
        Args:
            images (torch.Tensor): Image input tensor.
            img_masks (torch.Tensor): Image mask tensor.
            lang_tokens (torch.Tensor): Language token tensor.
            lang_masks (torch.Tensor): Language mask tensor.

        Returns:
            tuple: Tuple containing the output tensors.
        """
        images, img_masks, lang_tokens, lang_masks = self.process_next_obs(
            images, img_masks, vqh_images, vqh_img_masks, lang_tokens, lang_masks
        )

        embs, pad_masks, att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        suffix_out = self.vqh_backbone.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=embs,
        )  # (2B, S, E)

        batch_indices = torch.arange(suffix_out.shape[0], device=suffix_out.device)
        query_embedding_idx = pad_masks.sum(-1).long() - 1
        query_embedding = suffix_out[batch_indices, query_embedding_idx]

        cal_ql_batch: at.CalQlBatch = dict(
            encoded_observations=query_embedding[
                : int(query_embedding.shape[0] / 2)
            ].to(dtype=torch.float32),
            encoded_next_observations=query_embedding[
                int(query_embedding.shape[0] / 2) :
            ].to(dtype=torch.float32),
            actions=actions.view(actions.shape[0], -1),
            rewards=rewards,
            mc_returns=mc_returns,
            masks=masks,
        )
        temperature_loss, policy_loss, critic_loss, log_dict = self.calql(cal_ql_batch)

        return temperature_loss, policy_loss, critic_loss, log_dict

    @jaxtyped(typechecker=typechecker)
    def select_q_actions(
        self,
        images: list[Float[Tensor, "Batch 3 224 224"]],
        img_masks: list[Bool[Tensor, " Batch"]],
        lang_tokens: Int64[Tensor, "Batch seq_len"],
        lang_masks: Bool[Tensor, "Batch seq_len"],
        noise_actions: Float[
            Tensor,
            "Batch s2_candidates_num {self.config.vqh_chunk_size} {self.config.action_feature.shape[0]}",
        ],
    ) -> tuple[Int64[Tensor, " Batch"], Float[Tensor, "Batch s2_candidates_num"]]:
        batch_size = noise_actions.shape[0]
        s2_candidates_num = noise_actions.shape[1]
        embs, pad_masks, att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        suffix_out = self.vqh_backbone.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=embs,
        )  # (B, S, E)

        batch_indices = torch.arange(suffix_out.shape[0], device=suffix_out.device)
        query_embedding_idx = pad_masks.sum(-1).long() - 1
        query_embedding = suffix_out[batch_indices, query_embedding_idx]

        noise_actions = noise_actions.reshape(batch_size, s2_candidates_num, -1)
        q_values = self.calql.get_q_values(query_embedding, noise_actions)

        action_index = torch.argmax(q_values, dim=1)

        print(f"MaxValues: {q_values.max(dim=1)[0].tolist()}")
        print(f"MinValues: {q_values.min(dim=1)[0].tolist()}")
        print(f"MeanValues: {q_values.mean(dim=1)[0].tolist()}")
        print(f"ActionIndex: {action_index.tolist()}")

        return action_index, q_values
