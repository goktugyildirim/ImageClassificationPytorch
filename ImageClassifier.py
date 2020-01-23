#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
import sys
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd #data processing
import warnings
import matplotlib.image as mpimg
import torch
warnings.filterwarnings('ignore')
numpy.set_printoptions(threshold=sys.maxsize) #full print setting
#--------------------------------------------------------------------------------------
import torch
import PIL
from PIL import Image
import os



def DownSampling_Save(downsampling_ratio,folder_adress):
    i = 0
    
    data = []
    
    data = np.array(data)
    
    for filename in os.listdir(folder_adress):
        
        i=i+1
        
        image = Image.open(os.path.join(folder_adress,filename)).convert('L')
    
        print(image)
        
        height, width = image.size
        
        width = int(width/downsampling_ratio)
        
        height = int(height/downsampling_ratio)
        
        image = image.resize((width,height), Image.LANCZOS) #LANCZOS is downsampling filter.
        
        image = image.save(str(i)+".jpg","JPEG")
        
        image = np.array(image)
        
        print("x{} Downsampling, New shape = ".format(downsampling_ratio))
        print(image.shape)
        
        print("\n\n")
        
        
        
downsampling_ratio = 4

image_path = "C:/Users/Asus/Desktop/MSS-Project/cell_images/Uninfected"

DownSampling_Save(downsampling_ratio, image_path) 


# In[6]:


#Settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
import sys
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd #data processing
import warnings
import matplotlib.image as mpimg
import torch
warnings.filterwarnings('ignore')
numpy.set_printoptions(threshold=sys.maxsize) #full print setting
#--------------------------------------------------------------------------------------
import torch

def makeBatch(input, output, channel, batchSize):
    #This function takes  as input 4D array (numberOfSample, channel,input.shape[2](m), input.shape[3](n)) then returns
    # 5D array (batchIndex, batchSize, channel, input.shape[2](m), input.shape[3](n))
    
    numberOfSample = input.shape[0]
    channel = input.shape[1]
    m = input.shape[2]
    n = input.shape[3]
    
    batchIndex = int(numberOfSample/batchSize)
    
    x = np.ones((batchIndex,batchSize, channel,m,n))
    y = np.ones((batchIndex,batchSize,1))
        
    #result = np.zeros(batchIndex, batchSize, channel, m, n)
    
    for i in range(batchIndex):
        
        x[i] = input[i*batchSize:(i+1)*batchSize][:][:][:]
        y[i] = output[i*batchSize:(i+1)*batchSize][:]

    return x,y

def zeroPadding(image,square_dimension): 
    
    m=image.shape[0]
    n=image.shape[1]
    
    if (square_dimension-n)%2==0:
        
        x=int((square_dimension-n)/2)
        zero=np.zeros(shape = (m, x) )
        image = np.concatenate( (zero, image ), axis=1 )
        image = np.concatenate( (image,zero ), axis=1 )
        
    else:
        
        x=int(((square_dimension-n)-1)/2)
        y=int(((square_dimension-n)-1)/2)+1
        
        zero=np.zeros(shape = (m, x) )
        image = np.concatenate( (zero, image ), axis=1 )
        
        zero=np.zeros(shape = (m, y) )
        image = np.concatenate( (image,zero ), axis=1 )
        
    m=image.shape[0]
    n=image.shape[1]
    
    if (square_dimension-m)%2==0:
        
        x=int((square_dimension-m)/2)
        zero=np.zeros(shape = (x, n) )
        image = np.concatenate( (zero, image ), axis=0 )
        image = np.concatenate( (image,zero ), axis=0 )
        
    else:
        
        x=int(((square_dimension-m)-1)/2)
        y=int(((square_dimension-m)-1)/2)+1
        
        zero=np.zeros(shape = (x,n) )
        image = np.concatenate( (zero, image ), axis=0 )
        
        zero=np.zeros(shape = (y,n) )
        image = np.concatenate( (image,zero ), axis=0 )
        
    return image

def dataPreparing(square_dimension,lower_limit, folder_adress):

    data = []

    files = glob.glob (folder_adress)
    
    #counter=0;
    
    for myFile in files:
        #print(myFile)
        
        image=cv2.imread(myFile,0); #parameter=0 to single dimension image
    
        #Utilized infected images = 
        if image.shape[0]<square_dimension and image.shape[1]<square_dimension and image.shape[0]>lower_limit and image.shape[1]>lower_limit:
        
            image=zeroPadding(image,square_dimension)
        
            data.append(image)
            
    numpy_data=np.array(data)#making numpy array
    
    print(numpy_data.shape)
    
    return numpy_data



#**************************************************************************************************************************
class_1 = "C:/Users/Asus/Desktop/MSS-Project/cell_images/Parasitized x4/*.jpg" # 1
class_2 = "C:/Users/Asus/Desktop/MSS-Project/cell_images/Uninfected x4/*.jpg" # 0


lower_limit = 1
upper_limit = 40
channel = 1
batch = False
Normalization = False

if batch==True:
    batchSize = 20
    split_ratio = 0.5 #Batchli forward propagationda validation set ile training setin boyu aynı olmalı.
else:
    split_ratio = 0.15

#**************************************************************************************************************************
infected_data = dataPreparing(upper_limit,lower_limit, class_1) # 1

uninfected_data = dataPreparing(upper_limit,lower_limit, class_2) # 0

x_data = np.concatenate( (infected_data  , uninfected_data ), axis=0 )
infected_y = np.ones(infected_data.shape[0])
uninfected_y = np.zeros(uninfected_data.shape[0]) # yatay matrix
y_data = np.concatenate((infected_y, uninfected_y), axis=0).reshape(x_data.shape[0],1)


print("\nx_data shape: ",x_data.shape)
print("y_data shape: ",y_data.shape, end="\n")

#Data Split Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size= split_ratio, random_state=42)
number_of_train_X = X_train.shape[0]
number_of_test_X = X_test.shape[0]
number_of_train_Y = Y_train.shape[0]
number_of_test_Y = Y_test.shape[0]

#Flatten Image
x_train= X_train
x_test= X_test 
y_train = Y_train
y_test = Y_test


#y_train = np.concatenate(y_train)

x_train = np.expand_dims(x_train, axis=1)
x_train = np.tile(x_train, (1,1,1,1))

x_test = np.expand_dims(x_test, axis=1)
x_test = np.tile(x_test, (1,1,1,1))

print("--------------------------------------------------------\nx train shape without batching: {}".format(x_train.shape))
print("y train shape: without batching: {}".format(y_train.shape))

print("x validation shape without batching: {}".format(x_test.shape))
print("y validation shape without batching: {} \n--------------------------------------------------------".format(y_test.shape))

if(batch==True):
    x_train, y_train = makeBatch(x_train, y_train, channel, batchSize)
    x_test, y_test = makeBatch(x_test, y_test, channel, batchSize)
    print("x train shape after batching: {}".format(x_train.shape))
    print("y train shape: after batching: {}".format(y_train.shape))
    print("x validation shape after batching: {}".format(x_test.shape))
    print("y validation shape after batching: {} \n-------------------------------------------------------- ".format(y_test.shape))

x_train = torch.from_numpy(x_train).float()
x_validation = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_validation = torch.from_numpy(y_test).float()

#Data preparation is done.


# In[18]:


from IPython.core.debugger import set_trace # debug
import torch.nn as nn
import torch.optim as optimizer
#-----------------------------------
#Settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
import sys
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd #data processing
import warnings
import matplotlib.image as mpimg
import torch
warnings.filterwarnings('ignore')
numpy.set_printoptions(threshold=sys.maxsize) #full print setting
from sklearn.datasets import make_regression

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def prediction(x_validation):
    prediction = model(x_validation)
    thresholdPrediction = torch.zeros((prediction.shape[0],1))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(prediction.shape[0]):
        if prediction[i,0]<= 0.5:
            thresholdPrediction[i,0] = 0
        else:
            thresholdPrediction[i,0] = 1
            
    return thresholdPrediction

if(Normalization == True):
    x_train=x_train/255
    y_train=y_train
    x_validation = x_validation/255
    y_validation = y_validation

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        channel = 5
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = channel, kernel_size = 5) 
        #torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3)
        #torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(in_channels = channel, out_channels =channel, kernel_size = 3)
        #torch.nn.init.xavier_uniform(self.conv3.weight)
        #self.conv4 = torch.nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3)
        #torch.nn.init.xavier_uniform(self.conv4.weight)
        #self.conv5 = torch.nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5)
        #self.conv6 = torch.nn.Conv2d(in_channels = 5, out_channels = 5, kernel_size = 5)
        
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.dropout2 = nn.Dropout2d(0.5)
        
        self.pool = nn.MaxPool2d(2, 2)
    
        self.beta1 = nn.Linear(channel*3*3, 3)
        #torch.nn.init.xavier_uniform(self.beta1.weight)
        self.beta2 = nn.Linear(3, 10)
        #torch.nn.init.xavier_uniform(self.beta2.weight)
        self.beta3 = nn.Linear(10, 1)
        #torch.nn.init.xavier_uniform(self.beta3.weight)
        
    def forward(self, X):
        
        channel=5
       
        y=self.conv1(X) 
        
        y=torch.nn.functional.relu(y)
        y=self.pool(y) 
        y=self.conv2(y)
        y=torch.nn.functional.relu(y)
        y=self.pool(y) 
        y=self.conv3(y)
        y=torch.nn.functional.relu(y)
        y=self.pool(y)
        """y=self.conv4(y)
        y=torch.nn.functional.relu(y)
        y=self.pool(y)"""
        
        
        y= self.dropout1(y)
        
        #print(y.shape)
        
        y = y.view(X.shape[0], channel*3*3)
        
        y=self.beta1(y)
        y=torch.nn.functional.relu(y)
        y=self.beta2(y)
        y=torch.nn.functional.relu(y)
        y= self.dropout2(y)
        y=self.beta3(y)
        y=torch.nn.functional.sigmoid(y)
        
        return y
#***************************************************************************************************************************

loadParameters = False # True ise eğitime geçmişte eğitilen parameterlerle başlar.
learning_rate = 0.001
epoch = 1000
loadingRate = 10 # Loading and making prediction at epoch every loadParameters times 

if(loadParameters == True):
    model = model = load_checkpoint('Epoch 2500 Checkpoint.pth') #Geçmişte eğitilen bir model üzerinden forward propagation yapılır.
else:
    model = model = Net()
    
loss_fn = nn.MSELoss() #loss function is defined
#optimizer = optimizer.SGD(model.parameters(), lr= learning_rate) #optimizer is defined
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#***************************************************************************************************************************

epoch_list = []
validation_loss_list = []
training_loss_list = []
accuracy_list = []


print("\nTraining...\n")

try: 
    for i in range(epoch):

        if(batch==True):

            for k in range(x_train.shape[0]):

                epoch_list.append(i)

                model.train()

                optimizer.zero_grad() #optimizer gradientleri sıfırlandı.

                y_prediction = model(x_train[k][:][:][:][:]) #tahmin yapıldı
                loss = loss_fn(y_prediction, y_train[k][:][:]) #loss hesaplandı
                loss.backward() #türevler hesaplandı
                optimizer.step() #optimizer gradientleri güncelledi
                training_loss_list.append(loss.item())
                print("Epoch {} : Batch {} : Train Loss {}".format(i+1,k+1,loss.item()))

                # Eval
                model.eval()  # <-- here
                with torch.no_grad():
                    y_validation_prediction = model(x_validation[k][:][:][:][:])  
                loss = loss_fn(y_validation_prediction, y_validation[k][:][:])
                validation_loss_list.append(loss.item())
                print(" Validation Loss {}\n".format(loss.item()))

        if(batch==False):

            epoch_list.append(i)

            model.train()

            optimizer.zero_grad() #optimizer gradientleri sıfırlandı.

            y_prediction = model(x_train) #tahmin yapıldı
            loss = loss_fn(y_prediction, y_train) #loss hesaplandı
            loss.backward() #türevler hesaplandı
            optimizer.step() #optimizer gradientleri güncelledi
            training_loss_list.append(loss.item())
            print("[Epoch {}] : Training Loss: {}".format(i+1,loss.item()))

            # Eval
            model.eval()  # <-- here
            with torch.no_grad():
                y_validation_prediction = model(x_validation)  
            loss = loss_fn(y_validation_prediction, y_validation)
            validation_loss_list.append(loss.item())
            print("       Validation Loss: {}\n".format(loss.item()))
            
            #Prediction
            thresholdPrediction = prediction(x_validation)
            accuracy_list.append (100 - torch.mean(torch.abs(thresholdPrediction - y_validation)) * 100)
            print("\nTest Accuracy: {} %".format(100 - torch.mean(torch.abs(thresholdPrediction - y_validation)) * 100))
            print("\n")
            
            
        
        if(i%loadingRate==0 and i!=0): # Her loadingRate epochta bir kayıt yapılır.
            #ModelSaving
            checkpoint = {'model': Net(),
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}

            torch.save(checkpoint, 'Epoch {} Checkpoint.pth'.format(i))
            print("\nModel kaydedildi.\n")
            
            #Prediction
            thresholdPrediction = prediction(x_validation)
            print("\nTest Accuracy: {} %".format(100 - torch.mean(torch.abs(thresholdPrediction - y_validation)) * 100))
            print("\n")

            
            
except: #keyboard interrupt durumunda model kaydedilir.
       #ModelSaving
    checkpoint = {'model': Net(),
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint, 'Keyboard Interrupt Checkpoint.pth')
    print("\nModel kaydedildi.\n")

    
    print("Minimum training loss {} : Minimum validation loss {} ".format(min(training_loss_list),min(validation_loss_list)))

    del epoch_list[-1]
    fig, ax = plt.subplots()
    ax.plot(epoch_list, training_loss_list, label="Training Loss")
    ax.plot(epoch_list, validation_loss_list, label="Validation Loss")
    ax.set_title("Loss ")
    ax.legend();

    fig, ax = plt.subplots()
    ax.plot(epoch_list, accuracy_list)
    ax.set_title("Accurcacy ")
    ax.legend();
    
    
#Prediction
thresholdPrediction = prediction(x_validation)
print("\nTest Accuracy: {} %".format(100 - torch.mean(torch.abs(thresholdPrediction - y_validation)) * 100))

#ModelSaving
checkpoint = {'model': Net(),
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'Epoch {} Checkpoint.pth'.format(epoch))
print("\nModel kaydedildi.\n")

print("Minimum training loss {} : Minimum validation loss {}\nMaximum Accuracy {}".format(min(training_loss_list),min(validation_loss_list),max(accuracy_list)))
fig, ax = plt.subplots()
ax.plot(epoch_list, training_loss_list, label="Training Loss")
ax.plot(epoch_list, validation_loss_list, label="Validation Loss")
ax.set_title("Loss ")
ax.legend();

fig, ax = plt.subplots()
ax.plot(epoch_list, accuracy_list)
ax.set_title("Accurcacy ")
ax.legend();





# In[20]:


len(epoch_list)


# In[21]:


len(training_loss_list)


# In[22]:


len(validation_loss_list)


# In[23]:


len(accuracy_list)


# In[ ]:





# In[53]:





# In[25]:


fig, ax = plt.subplots()
ax.plot(epoch_list, accuracy_list)
ax.set_title("Accurcacy ")
ax.legend();


# In[50]:


path = "C:/Users/Asus/Desktop/MSS-Project/cell_images/TestDemonstration/P1.png"

def testPrediction(path):
   
    
    def zeroPadding(image,square_dimension): 
    
        m=image.shape[0]
        n=image.shape[1]

        if (square_dimension-n)%2==0:

            x=int((square_dimension-n)/2)
            zero=np.zeros(shape = (m, x) )
            image = np.concatenate( (zero, image ), axis=1 )
            image = np.concatenate( (image,zero ), axis=1 )

        else:

            x=int(((square_dimension-n)-1)/2)
            y=int(((square_dimension-n)-1)/2)+1

            zero=np.zeros(shape = (m, x) )
            image = np.concatenate( (zero, image ), axis=1 )

            zero=np.zeros(shape = (m, y) )
            image = np.concatenate( (image,zero ), axis=1 )

        m=image.shape[0]
        n=image.shape[1]

        if (square_dimension-m)%2==0:

            x=int((square_dimension-m)/2)
            zero=np.zeros(shape = (x, n) )
            image = np.concatenate( (zero, image ), axis=0 )
            image = np.concatenate( (image,zero ), axis=0 )

        else:

            x=int(((square_dimension-m)-1)/2)
            y=int(((square_dimension-m)-1)/2)+1

            zero=np.zeros(shape = (x,n) )
            image = np.concatenate( (zero, image ), axis=0 )

            zero=np.zeros(shape = (y,n) )
            image = np.concatenate( (image,zero ), axis=0 )

        return image 
    
    def prediction(x_validation):
        prediction = model(x_validation)
        thresholdPrediction = torch.zeros((prediction.shape[0],1))
        # if z is bigger than 0.5, our prediction is sign one (y_head=1),
        # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
        for i in range(prediction.shape[0]):
            if prediction[i,0]<= 0.5:
                thresholdPrediction[i,0] = 0
            else:
                thresholdPrediction[i,0] = 1

        return thresholdPrediction
    
    image = Image.open(path).convert('L')
    image.show()
    height, width = image.size
    width = int(width/downsampling_ratio)
    height = int(height/downsampling_ratio)
    image = image.resize((width,height), Image.LANCZOS) #LANCZOS is downsampling filter.
    image = np.array(image)
    image = zeroPadding(image,40) 
    image = np.expand_dims(image, axis=-3)
    image = np.tile(image, (1,1,1,1))
    image = torch.from_numpy(image).float() #numpy to pytorch
    thresholdPrediction = prediction(image)
    
    return  thresholdPrediction

testPrediction(path)


# In[ ]:




