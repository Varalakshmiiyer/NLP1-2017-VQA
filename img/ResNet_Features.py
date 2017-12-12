import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
from torchvision import transforms

import numpy
from PIL import Image

use_cuda = torch.cuda.is_available()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        cnn = resnet.resnet152(pretrained=True)
        self.modifiedCNN = nn.Sequential(*list(cnn.children())[:-1])

        # Using the imagenet mean and std
        self.transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(( 0.485, 0.456, 0.406 ),
                                 ( 0.229, 0.224, 0.225 ))])

    def forward(self, img_batch):
        """ Given an image path, this function will return the ResNet features """
        # TODO: Do preprocessing outside forward()

        batch_len = len(img_batch)
        Img_batch_tensor = torch.zeros([len(img_batch), 3, 224, 224])

        for idx, img_p in enumerate(img_batch):
            img = Image.open(img_p)
            if len(numpy.array(img).shape) == 2:
                rgb = Image.new('RGB',img.size)
                rgb.paste(img)
                img = rgb
                # print(img_p)
                # print(numpy.array(img).shape)
            img_tensor = self.transform(img)
            img_tensor.unsqueeze_(0)
            Img_batch_tensor[idx] = img_tensor

        if use_cuda:
            img = Variable(Img_batch_tensor, volatile=True).cuda()
        else:
            img = Variable(Img_batch_tensor, volatile=True)

        img_features = self.modifiedCNN(img)

        return img_features
