import tensorflow as tf
import tf2onnx
import onnx
from onnx2pytorch import ConvertModel
import torch
import os

def model2pth(h5_path, pth_path):
    """ 将 Keras 模型转换为 PyTorch 模型 """
    # 加载 Keras 模型
    keras_model = tf.keras.models.load_model(h5_path)
    # 转换为 ONNX 并保存
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, opset=13)
    # 从pth_path的同级目录下获取onnx_path
    onnx_path = pth_path.replace('.pth', '.onnx')
    onnx.save(onnx_model, onnx_path)
    # 转换为 PyTorch 并保存
    onnx_model = onnx.load(onnx_path)
    pytorch_model = ConvertModel(onnx_model)
    torch.save(pytorch_model.state_dict(), pth_path)
    print("模型转换完成。PyTorch模型已保存至:", pth_path)
    # 删除中间文件
    os.remove(onnx_path)



if __name__ == '__main__':
    h5_model_path = r'C:\Users\TLY\Desktop\DeepSample\DeepSample-Replication\data\model\modelA.h5'
    pytorch_model_path = r'C:\Users\TLY\Desktop\DeepSample\DeepSample-Replication\data\model\modelA.pth'
    model2pth(h5_model_path, pytorch_model_path)


