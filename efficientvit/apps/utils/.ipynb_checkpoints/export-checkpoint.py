import jittor as jt
from jittor import nn
import os

def export_jittor_model(model: nn.Module, export_path: str, sample_inputs: jt.Var):
    """Jittor原生导出方法 (实验性)"""
    # 1. 保存模型参数
    model.save(export_path + ".params")
    
    # 2. 保存计算图
    with jt.no_grad():
        output = model(sample_inputs)
        jt.save(output, export_path + ".output")
    
    # 3. 生成元数据
    meta = {
        "input_shape": sample_inputs.shape,
        "output_shape": output.shape,
        "model_class": model.__class__.__name__
    }
    jt.save(meta, export_path + ".meta")