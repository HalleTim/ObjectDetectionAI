import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50
import cv2

#!pip install faster-coco-eval
from tqdm.notebook import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection import IntersectionOverUnion

##################################
#Pfade der Ordnerstruktur anlegen#
##################################

path_trainData='Inputdata/train'
path_testData='Inputdata/test'
path_validData='Inputdata/valid'

path_train_output='OutputData/train'
path_test_output='OutputData/test'

path_infoData='Inputdata/data.yaml'

######################
#Parameter definieren#
######################

device="cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if device == "cuda" else False

lr_rate=1e-4
num_epochs=20
batch_size=32

#Skalierte Größe der Bilder defineren
scale_width=224
scale_height=224

#Gewichtung der Labels und der Markierung bei der Ausgabe der Ergebnisse
factLABELS = 1.0
factBBOX = 1.0

#Klasse für die Erstellung des Datensatzes
class CarDataSet(Dataset):
    def __init__(self, data, transforms=None):
        self.tensors=data
        self.transforms=transforms
       
    
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, index):
        image=self.tensors[0][index]
        label=self.tensors[1][index]
        bbox=self.tensors[2][index]
        
        image=image.permute(2,0,1)
        
        if self.transforms:
            image=self.transforms(image)
        
        return (image,label,bbox)
    

#einlesen der Kategorien
with open(path_infoData,'r') as file:
    infoData=yaml.safe_load(file)
    
ImageLabels=infoData['names']

simple_transform = transforms.Compose([transforms.Resize((scale_height,scale_width),antialias=True),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#Einlesen der Daten aus dem übergebenen Pfad und Erstellen des Datensatzes
def CreateDatasetFromPath(path):
    data=[]
    InsertDict={}
    LabelPath=path+"/labels/"
    ImagePath=path+"/images/"
    
    ImageLabel=[]
    imgData=[]
    ImageBboxes=[]
    for file in os.listdir(LabelPath):
        try:
            idx=file.split(".txt")[0]
            LabelFile=open(LabelPath + file,'r')
            lines=LabelFile.readlines()
            bboxes=[]
            
            for line in lines:
                LabelInfos= line.split(" ")
                
                
                startX=float(LabelInfos[1].strip())
                startY=float(LabelInfos[2].strip())
                EndX=float(LabelInfos[3].strip())
                EndY=float(LabelInfos[4].strip())
                
                bboxes.append((startX,startY,EndX,EndY))
            
            image=cv2.imread(ImagePath+idx+".jpg")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (scale_width, scale_height))
            
            
            imgData.append(image)
            ImageLabel.append(float(LabelInfos[0]))
            ImageBboxes.append(bboxes[0])
            
        except:
            print("Random Insertion Error")
    
    ImageLabel=torch.tensor(np.array(ImageLabel, dtype="int"))
    ImageBboxes=torch.tensor(np.array(ImageBboxes,dtype="float32"))
    imgData=torch.tensor(np.array(imgData, dtype="float32"))
    ImageLabel.to(device)
    ImageBboxes.to(device)
    imgData.to(device)

    return CarDataSet((imgData,ImageLabel,ImageBboxes), simple_transform)


#Einlesen Trainingsdaten
train_data=CreateDatasetFromPath(path_trainData)
#Einlesen Testdaten
test_data=CreateDatasetFromPath(path_testData)


#Umwandeln Prozentuale Angaben der Fahrzeugposition in absolute Werte
def convertToCoordinates(bbox):
    x_start, y_start = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2 
    
    x_start = int(x_start * scale_width) 
    y_start = int(y_start * scale_height)
    width = int(bbox[2] * scale_width)
    height = int(bbox[3] * scale_height)
    
    return x_start,y_start,width,height


train_Loader = DataLoader(train_data, batch_size=64,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)
test_Loader = DataLoader(test_data, batch_size=64,
	num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)


class ObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes=num_classes
    
        self.shared_layers=resnet50(weights="ResNet50_Weights.DEFAULT")

        self.regressor = nn.Sequential(
            nn.Linear(1000, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
            nn.Dropout(),
			nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
			nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
			nn.Linear(1000, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(256, self.num_classes)
		)

    def forward(self, x):
        shared_output=self.shared_layers(x)
        shared_output = shared_output.view(shared_output.size(0), -1)
        
        #Klassifikation in Autoklassen
        classification_output = self.classifier(shared_output)
        classification_output = torch.softmax(classification_output, dim=1)
        
        #Regression für die Markierung
        regression_output = self.regressor(shared_output)
        
        return classification_output, regression_output
    

def transformOutputToDict(labels,bboxes,calcIOU,*args):
    """Transformiert Tensor für weitere Verarbeitung mit Torchmertric in Liste aus Dictionary 
    oder berechnet Map Wert der Vorhersage. Verhalten Abhängig von Parameter calcMap(True/False)"""
    
    
    if(not calcIOU):
        #Transformation der Zielwerte für Torchmetric
        output_dict=[]
        for i in range(len(labels)):
            label=labels[i].tolist()
            bbox=bboxes[i].tolist()
            temp_dict={
                "boxes":torch.tensor([bbox]),
                "labels":torch.tensor([label])
            }
            output_dict.append(temp_dict)
        return output_dict
    else:
        #Transformation der Vorhersage und brechnung der IOU pro erfassten Image
        target=args[0]
        pred=[]
        iou=IntersectionOverUnion(box_format="xywh")
        for i in range(len(labels)):
            label=labels[i].tolist()
            bbox=bboxes[i].tolist()
            temp_dict={
                "boxes":torch.tensor([bbox]),
                "labels":torch.tensor([label])
            }
            score=iou([temp_dict],[target[i]])
            temp_dict["scores"]=torch.tensor([score["iou"].tolist()])
            pred.append(temp_dict)
        return pred
    
num_classes = len(ImageLabels)
detector=ObjectDetector(num_classes)


def plot_results(epochs, training_acc, testing_acc, training_loss, testing_loss):
    plt.plot(range(epochs), training_acc, label="train_acc")
    plt.plot(range(epochs), testing_acc, label="valid_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    plt.plot(range(epochs), training_loss, label="train_loss")
    plt.plot(range(epochs), testing_loss, label="valid_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def train_model(model, epochs=50, learning_rate=0.0001):
    #Optimierungverfahren festlegen
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
    #Verlustfunktion für Klassifikation
    ClassLoss_fn=nn.CrossEntropyLoss()
    #Verlustfunktion für Lokalisierung der Fahrzeuge
    BBoxLoss_fn=nn.MSELoss()
    
    training_loss = []
    testing_loss = []
    training_acc = []
    testing_acc = []
    #Listen um nach Abschluss des Trainings Map Wert bestimmen zu können
    metric = MeanAveragePrecision(box_format= 'xywh', iou_type="bbox")
    
    model=model.to(device)
    
    with tqdm(range(epochs)) as iterator:
        for epoch in iterator:
            pred=[]
            target=[]
###########################################################################
#################################Trainingsdaten############################
###########################################################################


            train_loss = 0
            train_acc = 0

            model.train()
            for images,labels,bbox in train_Loader:
                """if torch.cuda.is_available():
                    images = images.to("cuda")
                    labels = labels.to("cuda")
                    bbox = bbox.to("cuda")"""
                optimizer.zero_grad()
                output_classes, output_bbox=model(images)
                
                ClassLoss=ClassLoss_fn(output_classes,labels)
                BBoxLoss=BBoxLoss_fn(output_bbox,bbox)
                totalLoss=(factBBOX*BBoxLoss)+(factLABELS*ClassLoss)
                totalLoss.backward()
                optimizer.step()

                train_loss +=totalLoss.item()
                _,predicted=torch.max(output_classes,1)
                
                
                train_acc+=(predicted==labels).sum().item()
                                 
            training_acc.append(train_acc/len(train_data))
            training_loss.append(train_loss/len(train_data))
            
            
############################################################################
#################################Testdaten##################################
############################################################################


            test_loss = 0
            test_acc = 0
            with torch.no_grad():
                for images,labels,bbox  in test_Loader:
                    if torch.cuda.is_available():
                        images = images.to("cuda")
                        labels = labels.to("cuda")
                        bbox = bbox.to("cuda")
                    output_classes, output_bbox=model(images)
                    ClassLoss=ClassLoss_fn(output_classes,labels)
                    BBoxLoss=BBoxLoss_fn(output_bbox,bbox)
                    totalLoss=(factBBOX*BBoxLoss)+(factLABELS*ClassLoss)

                    _,predicted=torch.max(output_classes,1)
                    
                    
                    test_acc += (predicted == labels).sum().item()
                    test_loss += totalLoss.item()  
                    
                    #Dictionarys für den aktuellen Batch erstellen
                    Batch_target=transformOutputToDict(labels,bbox,False)
                    Batch_pred=transformOutputToDict(predicted,output_bbox,True,Batch_target)
                    
                    #Verknüpfung Batchdichtionary mit Liste für alle Testdaten um später Map bestimmen zu können
                    target=target+Batch_target
                    pred=pred+Batch_pred
                    
            
           
            
            testing_acc.append(test_acc/len(test_data))
            testing_loss.append(test_loss/len(test_data))
            iterator.set_postfix_str(f"train_acc: {train_acc/len(train_data):.2f} test_acc: {test_acc/len(test_data):.2f} train_loss: {train_loss/len(train_data):.2f} test_loss: {test_loss/len(test_data):.2f}")
    
    #Berechnung Map und ploten des Verlaufs von Accuracy und Loss
    metric.update(pred, target)
    print(metric.compute())
    plot_results(epochs, training_acc, testing_acc, training_loss, testing_loss)

train_model(detector)