import argparse
import PIL
import numpy as np
import math
import torch
from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models

def read_training_args():
    reader = argparse.ArgumentParser(description="Training")
    reader.add_argument('--gpu', action='store', default='gpu')
    reader.add_argument('--data_dir', action='store')
    reader.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    reader.add_argument('--epochs', dest='epochs', default=4)
    reader.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    reader.add_argument('--hidden_units', dest='hidden_units', default=512)
    reader.add_argument('--learning_rate', dest='learning_rate', default=0.001)
    return reader.parse_args()

def get_transformed_data(data_dir, data_type):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    if(data_type == 'train'):
        return datasets.ImageFolder(data_dir, transform=train_transforms)
    else:
        return datasets.ImageFolder(data_dir, transform=test_transforms)
    
def get_pretrained_model(arch):
    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_classifier(input_features, hidden_units):
    return nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
            ]))

def get_loss(inputs, labels, model, device, criterion, optimizer):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    return outputs, loss
    
def train_model(train_loader, validation_loader, model, device, criterion, optimizer, epochs):
    model.to(device)
    for e in range(int(epochs)):
        running_loss = 0
        step = 1
        for ii, (inputs, labels) in enumerate(train_loader):
            outputs, train_loss = get_loss(inputs, labels, model, device, criterion, optimizer)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            
            if(step%30 == 0):
                model.eval()
                with torch.no_grad(): validation_loss, accuracy = validate_model(validation_loader, model, device, criterion, optimizer)
                print("Epoch: {}/{} :\t\t".format(e+1, epochs),
                     "Training Loss: {:.4f} :\t\t".format(running_loss/30),
                     "Validation Loss: {:.4f} :\t\t".format(validation_loss/len(validation_loader)),
                     "Accuracy: {:.4f} :\t\t".format(accuracy))
                running_loss = 0
                
            step += 1
    return model

def validate_model(validation_loader, model, device, criterion, optimizer):
    model.to(device)
    correct = 0
    total = 50
    validation_loss = 0
    for ii, (inputs, labels) in enumerate(validation_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        validation_loss += loss
        prob = torch.exp(outputs) #tensor with prob. of each flower category
        pred = prob.max(dim=1) #tensor giving us flower label most likely

        #calculate number correct
        matches = (pred[1] == labels.data)
        correct += matches.sum().item()
        accuracy = 100*(correct/total)
    return validation_loss, accuracy

def test_model(test_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for ii, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test images :\t\t%d%%" % (100*correct/total))
    
def save_checkpoint(file_path, model, optimizer, args):
    checkpoint = {'args': args,
                  'model': model,
                  'optimizer': optimizer.state_dict(),
                  'classifier':model.classifier,
                  'state_dict': model.state_dict()
                 }
    torch.save(checkpoint, file_path)

#---------------------------------------------------------------------------------------------#
    
def read_prediction_args():
    reader = argparse.ArgumentParser(description="Predicting")
    reader.add_argument('--gpu', action='store', default='gpu')
    reader.add_argument('--top_k', dest='top_k', default='3')
    reader.add_argument('--checkpoint', action='store', default='checkpoint.pth')
    reader.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # use a deafault filepath to a primrose image 
    reader.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    return reader.parse_args()    

def load_checkpoint(file):
    return torch.load(file)

def get_input_test_image(image_path):
    image = PIL.Image.open(image_path)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = adjustments(image)
    return input_image

def predict(input_image, model, device, cat_to_name, top_k):
    top_k = int(top_k)
    model.eval()
    log_probs = model.forward(input_image)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(top_k)
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]+1
    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def print_probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {:.2f}%".format(j[1], j[0]*100))

