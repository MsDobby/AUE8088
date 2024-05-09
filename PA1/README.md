# PA1 : Image Classification

- [TODO](#❗TODO )
- [Model Zoo](#Model-Zoo)
- [Run](#Run)
- [Acknowledgement](#Acknowledgement)


# ❗TODO 
- [x] Accuracy at [`metric.py`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/metric.py#L53)
- [x] F1Score at [`metric.py`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/metric.py#L8)
- [x] Design my own network at [`network.py`]()
- [x] Toward SOTA at [`network.py`]() 

# Model Zoo

### Ablation Study 1 : Network Architecture
||F1 Score| val / loss|
|------|---|---|
|base|||
|global average pooling (GAP)|||
|my feature extractor|||
|GAP + my feature extractor|||

### Ablation Study 2 : Learning Algorithms

- Loss function
- Optimizer 

# Run
```bash
# If you want log to wandb,
python train.py --wandb

# If you DO NOT want log to wandb,
python train.py 
```


# Acknowledgement
You can see my work at this [link](https://wandb.ai/ophd/aue8088-pa1).
