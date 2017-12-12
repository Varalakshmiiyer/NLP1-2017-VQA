import os
import h5py
import pickle
import json
import torch
import numpy as np
from time import time

from ResNet_Features import CNN

use_cuda = torch.cuda.is_available()

def extract_img_features(dir_path, img_list, model):
    img_features = np.zeros((len(img_list), 2048))

    imgid2id = {}

    batch_size = 200
    batch_idx = 0
    flag = False
    img_path_list = [dir_path+'COCO_train2014_%012d.jpg'%img_id for img_id in img_list]

    while True:
        print(batch_idx)
        if (batch_idx+1)*batch_size < len(img_path_list):
            img_batch = img_path_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        else:
            flag = True
            img_batch = img_path_list[batch_idx*batch_size:]

        features = model(img_batch).squeeze()

        if use_cuda:
            features = features.cpu()
        features = features.data.numpy()

        for temp_idx, img in enumerate(img_batch):
            idx = batch_idx*batch_size+temp_idx
            imgid2id[img_list[idx]] = idx
            img_features[idx] = features[temp_idx]

        if flag:
            break

        batch_idx += 1

    return img_features, imgid2id


start = time()

img_dir = '/home/aashish/Documents/ProjectAI/data/MS_COCO/train2014/'

model = CNN()

if use_cuda:
    model.cuda()

print('Start')

with open('/home/aashish/Documents/TA/Project/data/image_ids_ir.json', 'r') as f:
    img_list = json.load(f)['image_ids']

img_features, imgid2id = extract_img_features(img_dir, img_list, model)

print('Extraction Done. Writing Files')

img_file = h5py.File('IR_image_features.h5', 'w')
img_file.create_dataset('img_features', dtype='float32', data=img_features)

json_data = {'IR_imgid2id':imgid2id}
with open('IR_img_features2id.json', 'a') as file:
    json.dump(json_data, file)
