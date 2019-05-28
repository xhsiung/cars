
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



idx = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(107648, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.size())
        x = x.view(-1, 107648)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc5(x)
        return x


model = Net()
model.load_state_dict(torch.load("char_recognizer.pt"))

def predict_char(gray):
    w = gray.size[0]
    h = gray.size[1]
    gray = gray.convert('L')
    gray = gray.point(lambda x: 0 if x<180 else 255, '1')
    x= int(64- (w/2))
    y = int(64- (h/2))
    canvas = Image.new('L', (128, 128), (255))
    canvas.paste(gray, box=(x, y))

    canvas = ImageOps.invert(canvas)
    canvas = np.array(canvas)
    canvas = canvas / 255.0
    
    #plt.imshow(canvas)
    #plt.show()

    #test_data = np.array(gray)
    test_output = model(Variable(torch.FloatTensor(canvas).unsqueeze(0).unsqueeze(0)))
    pred = test_output.data.max(1, keepdim=True)[1] 
    pred = np.array(pred).squeeze(0).squeeze(0)
    #print(idx[pred])
    return idx[pred]


def scaleImg(img2):
    img1=np.zeros((128,128,3),dtype=np.uint8)
    rows,cols= img2.shape 
    
    xx=64-int(rows/2)
    yy=64-int(cols/2)
    roi = img1[xx:rows+xx, yy:cols+yy]

    #img2gray = cv2.cvtColor( img2, cv2.COLOR_GRAY2RGB )
    ret,mask = cv2.threshold(img2 , 175, 255, cv2.THRESH_BINARY)
    mask_inv= cv2.bitwise_not(mask)

    img2 = cv2.cvtColor( img2, cv2.COLOR_GRAY2BGR)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[ xx:rows+xx, yy:cols+yy ] = dst
    
    return img1

#RGB
#pil_im =  Image.open("my1.png")
#predict_char(pil_im)

