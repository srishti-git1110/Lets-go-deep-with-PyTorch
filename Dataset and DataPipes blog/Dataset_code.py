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

class KaggleImageCaptioningDataset(Dataset):
  def __init__(self, train_captions, root_dir, transform=None, bert_model='distilbert-base-uncased', max_len=512):
    self.df = pd.read_csv(train_captions, header=None, sep='|')
    self.root_dir = root_dir
    self.transform = transform
    self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
    self.max_len = max_len

    self.images = self.df.iloc[:,0]
    self.captions = self.df.iloc[:,2]

  def __len__(self):
    return len(self.df)


  def __getitem__(self, idx):
    caption = self.captions[idx]
    image_id = self.images[idx]
    path_to_image = os.path.join(self.root_dir, image_id)
    image = Image.open(path_to_image).convert('RGB')
    
    if self.transform is not None:
      image = self.transform(image)

    tokenized_caption = self.tokenizer(caption, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.max_len,  
                                      return_tensors='pt')['input_ids']
    
    return image, tokenized_caption
  
  
  
root_dir = '/content/flickr30k-images-ecemod/image_dir'
train_captions = '/content/train_captions.csv'
bert_model = 'distilbert-base-uncased'
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.PILToTensor()])
train_dataset = KaggleImageCaptioningDataset(train_captions=train_captions,
                                       root_dir=root_dir,
                                       transform=transform,
                                       bert_model=bert_model)
train_loader = DataLoader(train_dataset, 
                          batch_size=64, 
                          num_workers=2, 
                          shuffle=True)

for batch_num, (image, caption) in enumerate(train_loader):
  if batch_num > 3:
    break
  print(f'batch number {batch_num} has {image.shape[0]} images and correspondingly {caption.shape[0]} tokenized captions')
