## Engine code

### Training
```Shell
cd UnLanedet
bash scripts/train.sh path_to_config num_gpus
# 8 cards training example 
# bash scripts/train.sh config/clrnet/resnet34_culane.py 8
```

### Evaluation
```Shell
cd UnLanedet
bash scripts/eval.sh path_to_config path_to_checkpoint
# 8 cards training example 
# bash scripts/eval.sh config/clrnet/resnet34_culane.py model_1000.path
```

### Resume
```Shell
cd UnLanedet
bash scripts/resume_train.sh path_to_config num_gpus
# 8 cards training example 
# bash scripts/resume_train.sh config/clrnet/resnet34_culane.py 8
```