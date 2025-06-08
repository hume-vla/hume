# LIBERO Evaluation
## Install
- Follow the instruction in `https://github.com/Lifelong-Robot-Learning/LIBERO`
```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
```
- Install dependences of Hume
```bash
pip install jaxtyping beartype tyro transformers==4.48.1 numpy==1.26.4
pip install git+https://github.com/huggingface/lerobot.git@768e36660d1408c71118a2760f831c037fbfa17d
``` 
## Download the Checkpoints
<table>
  <tr>
    <th>Model Name</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Hume-LIBERO-GOAL</td>
    <td><a href="https://huggingface.co/Hume-vla/Libero-Goal-1">hume-libero-goal-1</a></td>
  </tr>
</table>

## Evaluation
```
# Eval of LIBERO-SPATIAL
bash experiments/libero/scripts/libero_spatial.sh
# Eval of LIBERO-GOAL
bash experiments/libero/scripts/libero_goal.sh
# Eval of LIBERO-OBJECT
bash experiments/libero/scripts/libero_object.sh
# Evak of LIBERO-10
bash experiments/libero/scripts/libero_10.sh
```
> [!NOTE]
> We Provide serval `TTS args` in scripts, you can try them seperately 