import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
from model import HyperResNet34
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image

img_dir = './ColorClassification/'
filepath= './bestModel.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Env : {device}')


print('########################################## Train Data ##########################################')

filenames=[]
classes = []

with open('./ColorClassification/images.txt', 'r') as f:
  lines = f.readlines()
  
for line in lines:
  line = line.strip().split(' ')
  filenames.append(line[0])
  classes.append(int(line[1]))

datafiles = pd.DataFrame(
    {
        "images": filenames,
        "classes": classes
    }
)
print(datafiles.head(5))


print('########################################## Validation Data ##########################################')

validfilenames= []
validclasses = []

with open('./ColorClassification/timages.txt', 'r') as f:
  lines = f.readlines()
  
for line in lines:
  line = line.strip().split(' ')
  validfilenames.append(line[0])
  validclasses.append(int(line[1]))

validdatafiles = pd.DataFrame(
    {
        "images": validfilenames,
        "classes": validclasses
    }
)
print(validdatafiles.head(5))



class CustomData(Dataset):

  def __init__(self, datafiles, img_dir, transform=None):

    super().__init__()

    self.imgsfiles = datafiles
    self.img_dir = img_dir
    self.transform = transform

  def __len__(self, ):

    return len(self.imgsfiles)

  def __getitem__(self, idx):

    img_path = os.path.join(self.img_dir, self.imgsfiles.iloc[idx, 0])

    image = Image.open(img_path).convert('RGB')

    label = self.imgsfiles.iloc[idx, 1]

    if self.transform:
        image = self.transform(image)

    return image, label


# Transformation 

transform = T.Compose(
   [
    T.Resize((224, 224)),
    T.ToTensor(),
   ]
)

train = CustomData(datafiles, img_dir, transform)
trainloader = DataLoader(train, batch_size=32, shuffle=True)

valid = CustomData(validdatafiles, img_dir, transform)
validloader = DataLoader(valid, batch_size=32, shuffle=False)



def train(epoch):
  print(f'\nEpoch : {epoch}')

  model.train()

  train_loss=0
  correct=0
  total=0

  for img, label in tqdm(trainloader):

    img, label = img.to(device), label.to(device)

    optimizer.zero_grad()

    output = model(img) 

    loss = criterion(output, label)

    loss.backward()
    optimizer.step()

    train_loss += loss.item()

    total += label.size(0)

    _, predictions = output.max(1)
    correct += predictions.eq(label).sum().item()
      
  train_loss = train_loss/len(trainloader)
  
  accu = 100.*correct/total
  
  train_accu.append(accu)
  train_losses.append(train_loss)
  
  print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))


def validate(epoch):

  model.eval()

  running_loss=0
  correct=0
  total=0

  with torch.no_grad():
    for img, label in validloader:

      img, label = img.to(device), label.to(device)
      
      outputs = model(img)

      loss= criterion(outputs, label)

      running_loss += loss.item()

      total += label.size(0)

      _, predictions = outputs.max(1)
      correct += predictions.eq(label).sum().item()
  
  test_loss=running_loss/len(validloader)
  accu=100.*correct/total

  valid_losses.append(test_loss)
  valid_accu.append(accu)

  print('Validation Loss: %.3f | Accuracy: %.3f'%(test_loss, accu))

  return accu

if __name__ == '__main__':

	# Define the model

	model = HyperResNet34(9).to(device)

	# Setting up the model

	epochs = 250
	learning_rate = 1e-3
	optimizer = optim.SGD(model.parameters(), learning_rate)
	criterion = nn.CrossEntropyLoss()

	# Training the model 

	train_losses, train_accu = [], []
	valid_losses,  valid_accu = [], []
	maxAcc = 0
	for epoch in range(1, epochs):
	  train(epoch)
	  acc = validate(epoch)

	  if acc > maxAcc:
	    print('Save Best Model')
	    state = {
	    'epoch': epoch,
	    'state_dict': model.state_dict(),
	    'optimizer': optimizer.state_dict(),
	    }
	    torch.save(state, filepath)
	    maxAcc = acc