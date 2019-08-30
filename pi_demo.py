from armv7l.openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
import os
import time

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
t_prediction = 0

image = cv2.imread('/home/pi/Desktop/Expression5/1.jpg') 
img = cv2.resize(image.astype(np.float32), (44, 44))
img -= imagenet_mean
img = img.reshape((1, 44, 44, 3))
img = img.transpose((0, 3, 1, 2)) 

model_xml_CPU = '/home/pi/Desktop/Expression5/models/torch_model.xml'
model_bin_CPU = '/home/pi/Desktop/Expression5/models/torch_model.bin'

plugin = IEPlugin(device='MYRIAD') 
net = IENetwork(model=model_xml_CPU, weights=model_bin_CPU)
net.batch_size = 1 
input_blob = next(iter(net.inputs)) 
exec_net = plugin.load(network=net) 

t = time.time()
outputs = exec_net.infer(inputs={input_blob: img})
t_prediction = time.time() - t

class_name = class_names[np.argmax(outputs['617'])]
probs = outputs['617'][0, np.argmax(outputs['617'])]

print('Prediction time: %.2f' % t_prediction + ', Speed : %.2fFPS' % (1 / t_prediction))

print ("Class: " + class_name + ", probability: %.4f" %probs)






















# from armv7l.openvino.inference_engine import IENetwork, IEPlugin
# from data.fer import FER2013
# from torch.autograd import Variable
# import transforms as transforms
# import torch
# import numpy as np
# import time
# import torch.nn as nn

# model_xml_CPU = '/home/pi/Desktop/Expression5/models/torch_model.xml'
# model_bin_CPU = '/home/pi/Desktop/Expression5/models/torch_model.bin'

# plugin = IEPlugin(device='MYRIAD') 
# net = IENetwork(model=model_xml_CPU, weights=model_bin_CPU)


# transform_test = transforms.Compose([
#     transforms.RandomCrop(44),
#     transforms.ToTensor(),
# ])

# PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
# PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=1, shuffle=False, num_workers=1)

# correct = 0
# total = 0
# t_prediction = 0
# total_prediction_fps = 0

# criterion = nn.CrossEntropyLoss()

# for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
#     t = time.time()
#     print (inputs.shape)
#     test_bs, c, h, w = np.shape(inputs)
#     inputs = inputs.view(-1, c, h, w)
#     inputs, targets = inputs, targets
#     inputs, targets = Variable(inputs), Variable(targets)
#     print (1)
#     outputs = net(inputs)
#     print (2)
#     _, predicted = torch.max(outputs.data, 1)
#     t_prediction += (time.time() - t)
        
#     loss = criterion(outputs, targets)
#     PrivateTest_loss += loss.item()
#     total += targets.size(0)
#     correct += predicted.eq(targets.data).cpu().sum()

#     utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                        % (PrivateTest_loss / (batch_idx + 1), 100. *  float(correct) / float(total), correct, total))
# total_prediction_fps = (1 / (t_prediction / len(PrivateTestloader)))
# print('Prediction time: %.2f' % t_prediction + ', Average : %.5f/image' % (t_prediction / len(PrivateTestloader)) 
#       + ', Speed : %.2fFPS' % (1 / (t_prediction / len(PrivateTestloader))))
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
