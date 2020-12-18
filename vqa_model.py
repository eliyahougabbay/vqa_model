import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])     


import_or_install(datasets)
import_or_install(transformers)
import_or_install(sentencepiece)

import torch

from torch.utils.data import dataloader
from torch.utils.data import Dataset
from torchvision import models

import torchvision.transforms as transforms

from PIL import Image

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from typing import Any, Callable, Optional, Tuple



class VQADataset(Dataset):
  """
    This class loads a shrinked version of the VQA dataset (https://visualqa.org/)
    Our shrinked version focus on yes/no questions. 
    To load the dataset, we pass a descriptor csv file. 
    
    Each entry of the csv file has this form:
  

    question_id ; question_type ; image_name ; question ; answer ; image_id

  """
  def __init__(self, path : str, dataset_descriptor : str, image_folder : str, transform : Callable) -> None:
    """
      :param: path : a string that indicates the path to the image and question dataset.
      :param: dataset_descriptor : a string to the csv file name that stores the question ; answer and image name
      :param: image_folder : a string that indicates the name of the folder that contains the images
      :param: transform : a torchvision.transforms wrapper to transform the images into tensors 
    """
    super(VQADataset, self).__init__()
    self.descriptor = pd.read_csv(path + '/' + dataset_descriptor, delimiter=';')
    self.path = path 
    self.image_folder = image_folder
    self.transform = transform
    self.size = len(self.descriptor)
    
  
  def __len__(self) -> int:
    return self.size

  def __getitem__(self, idx : int) -> Tuple[Any, Any, Any]:
    """
      returns a tuple : (image, question, answer)
      image is a Tensor representation of the image
      question and answer are strings
    """
    image_name = self.path + '/' + self.image_folder + '/' + self.descriptor["image_name"][idx]

    image = Image.open(image_name).convert('RGB')

    image = self.transform(image)

    question = self.descriptor["question"][idx]

    answer = self.descriptor["answer"][idx]

    return (image, question, answer)


from torch.utils.data import DataLoader


# Précisez la localisation de vos données sur Google Drive
path = "/content/drive/MyDrive/Harispu-Sama"
image_folder = "boolean_answers_dataset_images_200"
descriptor = "boolean_answers_dataset_200.csv"

batch_size = 5

# exemples de transformations
transform = transforms.Compose(
    [transforms.Resize((299, 299)),
    transforms.ToTensor(),     
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ]
)
   

vqa_dataset = VQADataset(path, descriptor, image_folder, transform=transform)

vqa_dataset_train, vqa_dataset_test = torch.utils.data.random_split(vqa_dataset, [160, 40])

vqa_dataloader_train = DataLoader(vqa_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
vqa_dataloader_test = DataLoader(vqa_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)


from transformers import DistilBertTokenizer, DistilBertModel

tokenizer_distilbert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Model Text
distilbert = model_distilbert
for param in distilbert.parameters():
  param.requires_grad = False
distilbert.transformer.layer[5].output_layer_norm = torch.nn.Sequential(
    torch.nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True),
    # torch.nn.LSTM(input_size=768, hidden_size=1024, num_layers=2, dropout=.1, bidirectional=True, batch_first=True),
    torch.nn.Flatten(start_dim=1),
    torch.nn.Linear(in_features=13824, out_features=256),
    torch.nn.Sigmoid()
    )

# Model Image
inception = models.inception_v3(pretrained=True, aux_logits=False)
for param in inception.parameters():
  param.requires_grad = False
inception.fc = torch.nn.Sequential(
  torch.nn.Dropout(.05, inplace=False),
  torch.nn.Linear(in_features=2048, out_features=1024),
  torch.nn.Sigmoid(),
  torch.nn.Linear(in_features=1024, out_features=256),
  torch.nn.Sigmoid()
  )

import torch
import torch.nn.functional as F

class Text_model(torch.nn.Module):

  def __init__(self):
    super(Text_model, self).__init__()
    self.distilbert = distilbert
  
  def forward(self, x):
    return self.distilbert( **x )
    


class Image_model(torch.nn.Module):

  def __init__(self):
    super(Image_model, self).__init__()
    self.inception = inception
  
  def forward(self, y):
    # print(self.inception.fc[1].weight)
    return self.inception( y )



class VQA_model(torch.nn.Module):

  def __init__(self, text_model, image_model):
    super(VQA_model, self).__init__()

    self.text = text_model
    self.image = image_model

    self.sigmoid = torch.nn.Sigmoid()
    self.lin1 = torch.nn.Linear(in_features = 512, 
                                out_features = 256)
        
    self.last_layer = torch.nn.Linear(in_features = 256, 
                                      out_features =  1)
    
    self.dropout = torch.nn.Dropout(.2)

  
  def forward(self, image_batch, text_batch):
    # use x, y as respectively image and text
    x = self.image(image_batch)
    y = self.text(text_batch)[0]
    z = self.sigmoid( self.lin1( torch.cat( (x,y) , 1)) )
    # z = self.dropout( z )
    z = self.last_layer( z )
    z = self.sigmoid( z )

    return z

def train_optim(model, epochs, log_frequency, device, learning_rate=1e-4):

  model.to(device) # we make sure the model is on the proper device

  # Multiclass classification setting, we use cross-entropy
  # note that this implementation requires the logits as input 
  # logits: values prior softmax transformation 
  loss_fn = torch.nn.BCELoss(reduction='mean')

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  conversion = {'yes' : [1.], 'no' : [0.]}
  
  for t in tqdm(range(epochs)):

      model.train() # we specify that we are training the model

      # At each epoch, the training set will be processed as a set of batches
      for batch_id, batch in enumerate(vqa_dataloader_train) : 
        
          images, questions, answers = batch

          # ------ Preprocessing ------ #
          images = images.to(device)
          questions = tokenizer_distilbert(questions, return_tensors='pt', padding='max_length', add_special_tokens=False, max_length=18).to(device)
          answers = torch.Tensor(list(map(lambda key: conversion[key], answers))).to(device)

          # ---- computation model and loss ---#
          y_pred = model(images, questions) # forward pass output=logits
          # print('answers:{}  y_pred:{}\nanswers:{}  y_pred:{}\n'.format(answers[0], y_pred[0], answers[1], y_pred[1]))
          loss = loss_fn(y_pred, answers)

          if batch_id % log_frequency == 0:
              print("epoch: {:03d}, batch: {:03d}, loss: {:.3f} ".format(t+1, batch_id+1, loss.item()))

          optimizer.zero_grad() # clear the gradient before backward
          loss.backward()       # update the gradient
          optimizer.step() # update the model parameters using the gradient

        
      # Model evaluation after each step computing the accuracy
      model.eval()
      total = 0
      correct = 0
      for batch_id, batch in enumerate(vqa_dataloader_test):

          images, questions, answers = batch

          # ------ Preprocessing ------ #
          images = images.to(device)
          questions = tokenizer_distilbert(questions, return_tensors='pt', padding='max_length', add_special_tokens=False, max_length=18).to(device)
          answers = torch.Tensor(list(map(lambda key: conversion[key], answers))).to(device)

          y_pred = model(images, questions) # forward pass output=logits

          # print('y_pred {}'.format(y_pred))

          predicted = torch.round(y_pred)   # decision rule, we select the round
          
          total += answers.size(0)
          correct += (predicted == answers).sum().item()
        
      print("[validation] accuracy: {:.3f}%\n".format(100 * correct / total))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VQA_model(Text_model(), Image_model())

train_optim(model, epochs=100, log_frequency=10, device=device, learning_rate=0.005)
