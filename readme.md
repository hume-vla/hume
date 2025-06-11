<div align="center">

# Hume: Introducing System-2 Thinking in Visual-Language-Action Model



[\[📄Paper\]](https://arxiv.org/abs/2505.21432v1)  [\[🔥Project Page\]](https://hume-vla.github.io/) [\[📖 Document\]](#documents)

![perform](.assets/teaser_00.png)

</div>

## News
- `2025/06/08`: Release the evaluation code and ckpts on LIBERO
- `2025/06/03`: We release the training codebase of Hume.

## TODO LIST
- [x] Release Training Code
- [x] Release LIBERO Evaluation
- [ ] Using `uv` Manage Projects
- [ ] Release Real-Wold Evaluation
- [ ] Release Simpler Evaluation

## Try Hume
### LIBERO
> [!NOTE]
> Follow the instruction in [LIBERO](experiments/libero/README.md) 

## Documents

### 🚀 Quick Start

```bash
git clone https://github.com/hume-vla/hume.git
conda create -n hume python=3.10
conda activate hume
cd hume
pip install -e .
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
