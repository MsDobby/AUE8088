# PA1 : Image Classification

- [ToDo](#todo)
- [Performance Comparison](#performance-comparison)
- [Run](#run)
    + [train](#train)
    + [test](#test)
- [Acknowledgement](#acknowledgement)


# ToDo 
- [x] Accuracy at [`metric.py / MyAccuracy`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/metric.py#L53)
- [x] F1Score at [`metric.py / MyF1Score`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/metric.py#L8)
- [x] Try different settings at [`config.py`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/config.py)
- [x] Design my own network at [`network.py / MyNetwork`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/network.py#L137)
- [x] Toward SOTA at [`network.py / resnext18`](https://github.com/MsDobby/AUE8088-PA1/blob/master/PA1/src/network.py#L196)


# Performance Comparison
||acc / val | acc / train | train / loss | val / loss|
|------|---|---|---|---|
|AlexNet (base)|0.0201|0.2288|3.0155|3.1984|
|AlexNet + global average pooling|0.0238|0.3911|2.1718|3.0623|
|my feature extractor |**0.0390**|0.7652|**0.6487**|**1.6849**|
|ResNext-50 (SOTA)|0.0268|**0.8291**|0.6594|3.1556|

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
