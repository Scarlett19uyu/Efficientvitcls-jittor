# environment
RTX 3090 * 2卡
Jittor  1.3.1
Python  3.8(ubuntu18.04)
CUDA  11.3

# data
原数据集为ImageNet，共1000个分类，此处我们使用其子数据集，Imagenet100，在autodl有公共数据
```python
unzip /root/autodl-pub/ImageNet100/imagenet100.zip
```

# 训练脚本
```python
python applications/efficientvit_cls/train_efficientvit_cls_model.py \
applications/efficientvit_cls/configs/imagenet/default.yaml \
    --amp bf16 \
    --data_provider.data_dir /root/autodl-tmp/imagenet/ \
    --path efficientvit_cls/imagenet/efficientvit_b0_r224/
```

# 测试脚本
```python
python applications/efficientvit_cls/eval_efficientvit_cls_model.py \
applications/efficientvit_cls/configs/imagenet/default.yaml \
    --amp bf16 \
    --data_provider.data_dir /root/autodl-tmp/imagenet/ \
    --path efficientvit_cls/imagenet/efficientvit_b0_r224/
```
# pytorch版本
![image](https://github.com/user-attachments/assets/8e51cc9c-a5d8-419b-935b-c9a24f9aef64)

# jittor版本
![image](https://github.com/user-attachments/assets/1052ba3c-2964-430b-809e-1d7161dd25df)
