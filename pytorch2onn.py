from torch.autograd import Variable
import torch.onnx
import torchvision
from models import *

dummy_input = Variable(torch.randn(1, 3, 44, 44))
model = EdgeCNN()

model.load_state_dict(torch.load("/home/ysz/Mask_RCNN/data/pytoch/Expression5/models/FER2013_DenseNet/PrivateTest_model.t7"), strict=False)

torch.onnx.export(model, dummy_input, "/home/ysz/Mask_RCNN/data/pytoch/Expression5/torch_model.onnx", verbose=True)

print("Export of torch_model.onnx complete!")
