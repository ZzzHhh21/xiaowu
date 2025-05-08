
## Installation
Create a clean conda environment:
```
conda create -n tencent python=3.11
conda activate tencent
```

Then, install the required packages:
```
pip install -e .
```

## Training
```
python ./ppo/run_sb3_ppo_train.py 
python ./ppo/run_sb3_ppo_play.py --model ** 
```

## Citation
If you find HumanoidBench useful for your research, please cite this work:
```
@article{
}
```