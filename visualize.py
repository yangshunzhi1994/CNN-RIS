import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import cv2 as cv
import dlib
import time


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

def get_color(emotion, prob):
    if emotion.lower() == 'Angry':
        color = prob * np.asarray((0, 0, 255))
    elif emotion.lower() == 'Disgust':
        color = prob * np.asarray((255, 0, 0))
    elif emotion.lower() == 'Fear':
        color = prob * np.asarray((0, 255, 255))
    elif emotion.lower() == 'Happy':
        color = prob * np.asarray((255, 255, 0))
    elif emotion.lower() == 'Sad':
        color = prob * np.asarray((255, 255, 255))
    elif emotion.lower() == 'Surprise':
        color = prob * np.asarray((255, 0, 255))
    else:
        color = prob * np.asarray((0, 255, 0))
    return color

def draw_bounding_box(image, coordinates, color):
    x, y, w, h = coordinates
    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
def draw_text(image, coordinates, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv.putText(image, text, (x + x_offset, y + y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv.LINE_AA)

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

detector = dlib.get_frontal_face_detector()

net = EdgeNet()
checkpoint = torch.load(os.path.join('/home/pi/Desktop/Expression/models/FER2013_EdgeNet', 'PrivateTest_model.t7'),
                        map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['net'])
net.cpu()
net.eval()

cap = cv.VideoCapture(0)
while(1):
    start = time.time()
    # get a frame
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    # show a frame
    frame = frame[100:100 + width, :]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    faces = detector(gray, 1)
    
    for rect in faces:
        (x, y, w, h) = rect_to_bb(rect)
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
        gray_face = gray[y1:y2, x1:x2]
       
        img = cv.resize(gray_face, (44, 44))
        
        img = Image.fromarray(img)
        inputs = transform_test(img)

        c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)

        inputs = inputs.cuda()
        inputs = Variable(inputs)
        
        outputs = net(inputs)
        
        score = F.softmax(outputs)
        _, predicted = torch.max(outputs.data, 0)
        
        emotion = class_names[int(predicted.cpu().numpy())]
        prob = max(score.data.cpu().numpy())
        color = get_color(emotion, prob)
        
        
        text = emotion + '  ' + str(round(prob, 5))
        
        draw_bounding_box(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
        draw_text(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=text)
        cv.imshow("capture", frame) 
        
    end = time.time()
    seconds = end - start
    fps = 1.0 / seconds
    print('fps: %.2f' % fps)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows() 


























