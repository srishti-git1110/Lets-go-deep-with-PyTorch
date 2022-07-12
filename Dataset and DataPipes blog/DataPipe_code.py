!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d lefterislymp/neuralsntua-image-captioning
!unzip /content/neuralsntua-image-captioning.zip

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
import os
import pandas as pd

!pip install transformers
from transformers import AutoTokenizer

!pip install torchdata
import torchdata.datapipes as dp
from torch.utils.data.backward_compatibility import worker_init_fn
from torch.utils.data import DataLoader

training_csv = '/content/train_captions.csv'
train_dp = dp.iter.FileOpener([training_csv])
train_dp = train_dp.parse_csv(delimiter='|')
train_dp = train_dp.shuffle(buffer_size=2000)
train_dp = train_dp.sharding_filter()

max_len = 512
root_dir = '/content/flickr30k-images-ecemod/image_dir'

def apply_image_transforms(image):
  
  transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.PILToTensor()])
  return transform(image)

def open_image_from_imagepath(row):
  image_id, _, caption = row
  path_to_image = os.path.join(root_dir, image_id)
  image = Image.open(path_to_image).convert('RGB')
  image = apply_image_transforms(image)
  tokenized_caption = tokenizer(caption, 
                                padding='max_length',  # Pad to max_length
                                truncation=True,  # Truncate to max_length
                                max_length=max_len,  
                                return_tensors='pt')['input_ids']
  return {'image':image, 'caption':tokenized_caption}

  
train_dp = train_dp.map(open_image_from_imagepath)
train_loader = DataLoader(dataset=train_dp, shuffle=True, batch_size=32, num_workers=2, worker_init_fn=worker_init_fn)

num_epochs = 1
bert_model = 'distilbert-base-uncased'    # use any model of your choice
tokenizer = AutoTokenizer.from_pretrained(bert_model)
for epoch in range(num_epochs):
  for batch_num, batch_dict in enumerate(train_loader):
            if batch_num > 2:
                break
            
            images, captions = batch_dict['image'], batch_dict['caption']
            print(f'Batch {batch_num} has {images.shape[0]} images and correspondingly {captions.shape[0]} captions')
