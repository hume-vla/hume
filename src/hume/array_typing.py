from typing import Annotated, TypeAlias, TypedDict

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor


class InferConfig(TypedDict):
    """Configuration for inference."""

    replan_steps: int
    s2_replan_steps: int
    s2_candidates_num: int
    noise_temp_lower_bound: float
    noise_temp_upper_bound: float
    time_temp_lower_bound: float
    time_temp_upper_bound: float
    post_process_action: bool
    device: str


ImageArray: TypeAlias = Annotated[NDArray[np.uint8], "Shape[B, H, W, C]"]
StateArray: TypeAlias = Annotated[
    NDArray[np.float32], "Shape[B, state_horizon, state_dim]"
]
ActionArray: TypeAlias = Annotated[NDArray[np.float32], "Shape[B, action_dim]"]

InferBatchObs = TypedDict(
    "BatchObs",
    {
        "observation.images.image": ImageArray,
        "observation.images.wrist_image": ImageArray,
        "observation.state": StateArray,
        "task": list[str],
    },
)


class InferOutput(TypedDict):
    noise_action: Float[Tensor, "batch s2_chunksize padded_action_dim"]
    s1_action: Float[Tensor, "batch s1_chunksize unpadded_action_dim"]
    s2_action: Float[Tensor, "batch s2_chunksize unpadded_action_dim"]


class CalQlBatch(TypedDict):
    encoded_observations: Float[Tensor, "batch encoded_dim"]
    encoded_next_observations: Float[Tensor, "batch encoded_dim"]
    actions: Float[Tensor, "batch action_dim"]
    rewards: Float[Tensor, " batch"]
    mc_returns: Float[Tensor, " batch"]
    masks: Float[Tensor, " batch"]


class EnvArgs(TypedDict):
    """Environment arguments."""

    # necessary args
    num_trials_per_task: int
    num_steps_wait: int
    task_suite_name: str
    seed: int
    ckpt_path: str | None
    eval_name: str | None


class Request(TypedDict):
    """Environment receive message."""

    frame_type: str  # "init" | "action"
    env_args: EnvArgs | None
    action: ActionArray | None


class Response(TypedDict):
    """Environment send message."""

    status: str  # "new_episode" | "eval_finished" | "in_episode"
    success_rate: float | None
    observation: InferBatchObs | None
