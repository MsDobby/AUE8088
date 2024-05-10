# PA1 : Image Classification

- [ToDo](#todo)
- [Performance Comparison](#performance-comparison)
- [Optimize Hyper-params](#optimize-hyper-params)
- [Run](#run)
    + [Basic](#basic)
    + [Sweeps](#sweeps)
- [Acknowledgement](#acknowledgement)


# ToDo 
- [x] Accuracy at [`metric.py / MyAccuracy`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/metric.py#L53)
- [x] F1Score at [`metric.py / MyF1Score`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/metric.py#L8)
- [x] Try different settings at [`config.py`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/config.py)
- [x] Design my own network at [`network.py / MyNetwork`]()
- [x] Toward SOTA at [`network.py / resnext18`]() 

# Performance Comparison
||F1 Score| train / loss | val / loss|pretrained weights|
|------|---|---|---|---|
|AlexNet (base)|0.020125430077314377|3.0154902935028076|3.198350429534912|[tba.ckpt]()|
|AlexNet + global average pooling|0.023817744106054306|2.1718180179595947|3.062333583831787|[tba.ckpt]()|
|my feature extractor |0.03895244374871254|0.6486892700195312|1.684914231300354|[tba.ckpt]()|
|ResNext-50 (SOTA)||||[tba.ckpt]()|

# Run
### train
```bash
# If you want log to wandb,
python train.py --wandb

# If you DO NOT want log to wandb,
python train.py 
```
### test
```bash
python test.py --cfg_file ${file with extension ".ckpt"}
```


# Acknowledgement
You can see my work at this [link](https://wandb.ai/ophd/aue8088-pa1).
