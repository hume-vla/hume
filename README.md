<div align="center">

# Hume: Introducing System-2 Thinking in Visual-Language-Action Model



[\[📄Paper\]](https://arxiv.org/abs/2505.21432)  [\[🔥Project Page\]](https://hume-vla.github.io/) [\[📖 Document\]](#documents) [\[🤗 Model Zoo\]](https://huggingface.co/collections/Hume-vla/hume-model-zoo-684be6e5d062717593589a9a)

![perform](.assets/teaser_00.png)

</div>

## News
- `2025/06/12`: Manage the project dependency using `uv`. Isolate the evaluation and training environments
- `2025/06/08`: Release the evaluation code and ckpts on LIBERO.
- `2025/06/03`: We release the training codebase of Hume.

## TODO LIST
- [x] Release Training Code
- [x] Release LIBERO Evaluation
- [x] Switch to `uv` Manage the Project Dependency
- [ ] Release Real-Wold Evaluation
- [ ] Release Simpler Evaluation

## Try Hume
```python
from hume import HumePolicy
import numpy as np

# load policy
hume = HumePolicy.from_pretrained("/path/to/checkpoints").to("cuda")

# config Test-Time Computing args
hume.init_infer(
    infer_cfg=dict(
        replan_steps=8,
        s2_replan_steps=16,
        s2_candidates_num=5,
        noise_temp_lower_bound=1.0,
        noise_temp_upper_bound=1.0,
        time_temp_lower_bound=0.9,
        time_temp_upper_bound=1.0,
        post_process_action=True,
        device="cuda",
    )
)

# prepare observations
observation = {
    "observation.images.image": np.zeros((1,224,224,3), dtype = np.uint8), # (B, H, W, C)
    "observation.images.wrist_image": np.zeros((1,224,224,3), dtype = np.uint8), # (B, H, W, C)
    "observation.state": np.zeros((1, 7)), # (B, state_dim)
    "task": ["Lift the papper"],
}

# Infer the action
action = hume.infer(observation) # (B, action_dim)

```
### LIBERO
> [!NOTE]
> Follow the instruction in [LIBERO](experiments/libero/README.md) 

## Documents

### 🛠️ Installation
> We use [uv](https://docs.astral.sh/uv/getting-started/installation/) manage our project dependency
>
> Install `uv` in just one line command
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

```bash
git clone https://github.com/hume-vla/hume.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
```


### 🌟 **Train System2**
- Download pretrained System2 Weights <a href="https://huggingface.co/Hume-vla/Hume-System2">hume-system2-pt</a>
- Set `pretrained_policy` in `scripts/train_s2.sh` to path of the pretrained System2 weights
- Set environment veriable in `scripts/env.sh`
- Launch training
```bash
bash scripts/train_s2.sh
```

### 🌟 **Train System2 and Value Query Head**
- Set `pretrained_s2_path` in  `scripts/train_vqh_s1.sh`
- Download [DinoV2-Small](https://huggingface.co/facebook/dinov2-small) and set `--pretrained_dino_path` to the DinoV2-Small in `scripts/train_vqh_s1.sh`
- Launch training
```bash
bash scripts/train_vqh_s1.sh
```

## 🤗 Model Zoo

<table>
  <tr>
    <th>Model Name</th>
    <th>Link</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>Hume-System2</td>
    <td><a href="https://huggingface.co/Hume-vla/Hume-System2">hume-system2-pt</a></td>
    <td>pretrained weights exported from pi0 </td>
  </tr>
</table>

## 🎄 Datasets
- You can download all lerobot dataset used in our project from [here](https://huggingface.co/IPEC-COMMUNITY)
<table>
  <tr>
    <th>Dataset Name</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>LIBERO-Spatial</td>
    <td><a href="https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_lerobot">IPEC-COMMUNITY/libero_spatial_no_noops_lerobot</a></td>
  </tr>
  <tr>
    <td>LIBERO-Goal</td>
    <td><a href="https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_lerobot">IPEC-COMMUNITY/libero_goal_no_noops_lerobot</a></td>
  </tr>
  <tr>
    <td>LIBERO-Object</td>
    <td><a href="https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_lerobot">IPEC-COMMUNITY/libero_object_no_noops_lerobot</a></td>
  </tr>
  <tr>
    <td>LIBERO-10</td>
    <td><a href="https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_lerobot">IPEC-COMMUNITY/libero_10_no_noops_lerobot</a></td>
  </tr>
  <tr>
    <td>Bridge</td>
    <td><a href="https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot">IPEC-COMMUNITY/bridge_orig_lerobot</a></td>
  </tr>
  <tr>
    <td>Fractal</td>
    <td><a href="https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot">IPEC-COMMUNITY/fractal20220817_data_lerobot</a></td>
  </tr>
  
</table>

- If you want to train Hume using your own dataset, check the convert scripts in [any4lerobot](https://github.com/Tavish9/any4lerobot): 🚀 A collection of utilities and tools for LeRobot. ![GitHub Repo stars](https://img.shields.io/github/stars/Tavish9/any4lerobot)


## 🤗 FAQs
If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. We appreciate your feedback and contributions! 🚀

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{song2025hume,
  title={Hume: Introducing System-2 Thinking in Visual-Language-Action Model},
  author={Anonimous Authors},
  journal={arXiv preprint arXiv:2505.21432},
  year={2025}
}
```
