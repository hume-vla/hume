#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Callable

import datasets
import torch
import torch.utils
from datasets import load_dataset
from lerobot.common.constants import OBS_ROBOT
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.robot_devices.robots.utils import Robot

CODEBASE_VERSION = "v2.1"


class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        slide: int = 1,
        s1_action_steps: int = 5,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        wrist_transforms: Callable | None = None,
        discount: float = 0.98,
        num_pos: int = 3,
        next_obs_offset: int = None,
        s1_his_state_size: int = 1,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        self.wrist_transforms = wrist_transforms
        self.next_obs_offset = next_obs_offset

        # hume attributes
        self.slide = slide
        self.s1_action_steps = s1_action_steps
        self.s1_his_state_size = s1_his_state_size

        self._state_keys = [
            key for key in self.meta.features.keys() if key.startswith(OBS_ROBOT)
        ]
        self.s1_state_delta_indices = {
            key: [-i for i in range(0, s1_his_state_size)] for key in self._state_keys
        }

        # rewards
        self.discount = discount
        self.num_pos = num_pos
        self.reward_dataset = self.construct_reward_data()

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset(
                "parquet", data_dir=path, split="train", keep_in_memory=True
            )
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx))
                for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset(
                "parquet", data_files=files, split="train", keep_in_memory=True
            )
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def construct_reward_data(self):
        reward_keys = ["reward", "mc", "tdm", "done"]
        reward_data = {k: [] for k in reward_keys}

        from datasets import Dataset
        from tqdm import tqdm

        for i in tqdm(range(self.num_episodes)):
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()

            reward = torch.zeros(ep_end - ep_start, dtype=torch.float32)
            reward[: -self.num_pos] = -1.0
            td_mask = reward.bool()

            mc_ret = torch.zeros_like(reward)
            prev_ret = 0.0
            for t in range(len(reward) - 1, -1, -1):
                mc_ret[t] = reward[t] + self.discount * prev_ret * td_mask[t]

            done = torch.zeros(ep_end - ep_start, dtype=torch.bool)
            done[-1] = True

            for k, v in zip(reward_keys, [reward, mc_ret, td_mask, done]):
                reward_data[k].append(v)

        for k, v in reward_data.items():
            reward_data[k] = torch.cat(v, dim=0)
        reward_dataset = Dataset.from_dict(reward_data, split="train")
        reward_dataset.set_transform(hf_transform_to_torch)
        return reward_dataset

    def _get_s1_state_query_indices(
        self, idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [
                max(ep_start.item(), min(ep_end.item() - 1, idx + delta))
                for delta in delta_idx
            ]
            for key, delta_idx in self.s1_state_delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.s1_state_delta_indices.items()
        }
        return query_indices, padding

    def __len__(self):
        return self.num_frames * self.slide

    def __getitem__(self, idx) -> dict:
        s2_idx = idx // self.slide
        item = self.hf_dataset[s2_idx]
        ep_idx = item["episode_index"].item()

        ep_start = self.episode_data_index["from"][ep_idx].item()
        ep_end = self.episode_data_index["to"][ep_idx].item()

        delta_slide = idx % self.slide * self.s1_action_steps
        s1_idx = max(ep_start, min(ep_end - 1, s2_idx + delta_slide))
        s1_item = self.hf_dataset[s1_idx]

        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = (
                self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            )
            query_indices, padding = self._get_query_indices(s2_idx, current_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

            # query s1 video frames
            s1_query_timestamps = self._get_query_timestamps(
                s1_item["timestamp"].item()
            )
            s1_video_frames = self._query_videos(s1_query_timestamps, ep_idx)
        s1_item = {**s1_video_frames, **s1_item}

        # query s1 history state data
        s1_state_query_indices, s1_state_padding = self._get_s1_state_query_indices(
            s1_idx, current_ep_idx
        )
        s1_state_query_result = self._query_hf_dataset(s1_state_query_indices)

        # query vqh data
        offset = self.next_obs_offset or self.slide * self.s1_action_steps
        vqh_item = self.hf_dataset[min(s2_idx + offset, ep_end - 1)]

        vqh_query_timestamps = self._get_query_timestamps(vqh_item["timestamp"].item())
        vqh_video_frames = self._query_videos(vqh_query_timestamps, ep_idx)
        vqh_item = self.reward_dataset[s2_idx]
        vqh_item = {**vqh_item, **vqh_video_frames}

        # merge s1 s2 vqh items
        # __import__("ipdb").set_trace()
        merge_keys = list(filter(lambda k: k.startswith("observation."), item.keys()))
        for k in merge_keys:
            item[f"{k}.s1"] = s1_item[k]
        for k, v in s1_state_query_result.items():
            item[f"{k}.s1"] = v  # here, state in s1 is owerwritten
        for k, v in s1_state_padding.items():
            item[f"{k}.s1"] = v
        for k, v in vqh_item.items():
            item[f"{k}.vqh"] = v

        item["stamp"] = torch.tensor(
            [idx % self.slide / self.slide], dtype=torch.float32
        )
        item["s2_idx"] = s2_idx
        camera_keys = (
            self.meta.camera_keys
            + [f"{key}.s1" for key in self.meta.camera_keys]
            + [f"{key}.vqh" for key in self.meta.camera_keys]
        )
        if self.image_transforms is not None:
            image_keys = filter(lambda key: "wrist" not in key, camera_keys)
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        if self.wrist_transforms is not None:
            wrist_keys = filter(lambda key: "wrist" in key, camera_keys)
            for wrist in wrist_keys:
                item[wrist] = self.wrist_transforms(item[wrist])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        return item

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = super().create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            tolerance_s=tolerance_s,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            video_backend=video_backend,
        )
        obj.wrist_transforms = None
        return obj
