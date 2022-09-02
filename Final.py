#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
import os
import gc
import re
from lawreader import *
from itertools import *
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader, default_collate
from tqdm import tqdm
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, util
from utils import target_distribution, cluster_accuracy


# In[2]:


# Hyperparameters
THRESHOLD = 0.7
DEVICE = "cpu"
# torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
ToTrain = False


# In[3]:


# First model
Base_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
df = read_directory("data")
verdicts = dict()
for index in range(0,1,1):
    verdict = tidy(oread(df.iloc[index]["Path"], df.iloc[index]["Verdict"]))
    if verdict == []:
        verdicts[df.iloc[index]["Verdict"]] = ["本件為少年事件，依少年事件處理法第83條規定，未公開其內容。"]
        continue
    embeddings = Base_model.encode(verdict)
    dic = dict()
    for i in range(len(embeddings)):
        dic[verdict[i]] = embeddings[i]

    # Parameter setting
    S = -np.square(pairwise_distances(embeddings)) #Affinity matrix
    prefer = np.mean(S) #set preference value

    # Affinity Propagation method
    First_model = AffinityPropagation(preference = prefer)
    First_model.fit(embeddings)
    labels = First_model.fit_predict(embeddings)
    cluster_center = First_model.cluster_centers_

    # Grouped data
    labeled = list(zip(labels, dic.keys(), dic.values()))
    cluster = dict()
    for sentence in labeled:
        if cluster.__contains__(sentence[0]) == False:                
            cluster[sentence[0]] = []
        cluster[sentence[0]].append([sentence[1],sentence[2]])
    for i in range(len(cluster)):
        for sentence_pair in combinations(cluster[i], 2):
            similarity = util.pytorch_cos_sim(sentence_pair[0][1], sentence_pair[1][1]).item()
            if similarity >= THRESHOLD:
                similarity1 = util.pytorch_cos_sim(sentence_pair[0][1], cluster_center[i]).item()
                similarity2 = util.pytorch_cos_sim(sentence_pair[1][1], cluster_center[i]).item()
                if similarity1 == 1 or similarity2 == 1:
                    continue
                if similarity1 > similarity2:
                    try:
                        cluster[i].remove(sentence_pair[1])
                    except:
                        pass
                else:
                    try:
                        cluster[i].remove(sentence_pair[0])
                    except:
                        pass

    output = []
    for group in cluster.values():    
        for sentence, embeddings in group:
            output.append(sentence)
    if output != []:
        verdicts[df.iloc[index]["Verdict"]] = output         


# In[6]:


evidences = ex_evidence("evidences.txt")


# In[7]:


verdicts_and_evidences = VE_Dataset(verdicts, evidences)


# In[8]:


# Second model
class VE_Dataset(Dataset):
    # 讀取經first model處理後的判決書
    # 用於配對文句與證據
    def __init__(self, verdicts, evidences):
        assert type(verdicts) == dict, "Wrong input"
        assert type(evidences) == list, "Wrong input"
        self.verdict = []
        for name, verdict in verdicts.items():
            self.verdict.append((name, verdict))
        self.evidences = evidences
        self.len = len(verdicts)

    def __getitem__(self, idx):
        sentences = self.verdict[idx][1]
        evidence = self.evidences[idx]
        return sentences, evidence, len(evidence)
    
    def __len__(self):
        return self.len


# In[9]:


# Second model
for i in range(1):
    # K-means clustering
    ver, evidences, clusters_num = verdicts_and_evidences[i]
    embeddings = Base_model.encode(ver)
    evi_embeddings = Base_model.encode(evidences)
    Second_model = KMeans(n_clusters=clusters_num, random_state=46)
    Second_model.fit(embeddings)
    labels = Second_model.fit_predict(embeddings)
    cluster_center = Second_model.cluster_centers_
    labeled = list(zip(labels, ver))
    
    cluster = dict()
    for sentence in labeled:
        if cluster.__contains__(sentence[0]) == False:                
            cluster[sentence[0]] = []
        cluster[sentence[0]].append(sentence[1])
    
    # Assign each cluster to corresponding evidence:
    outcome = Second_model.predict(evi_embeddings)
    outcome = np.array([0,3,1,2])
    allocate = dict()
    for i in range(len(evidences)):
        x = outcome[i]
        allocate[x] = evidences[i]
    key_list = []
    for key, value in cluster.items():
        n_key = allocate[key]
        key_list.append([key, n_key])
    for key, n_key in key_list:
        cluster[n_key] = cluster.pop(key)
    
    sentence_to_evidence = cluster
    
    #results visualization
    plt.figure()
    plt.scatter(embeddings[:,0], embeddings[:,1], c = labels)
    plt.scatter(cluster_center[:,0], cluster_center[:,1], c = 'r')
    plt.axis('equal')
    plt.title('Prediction')
    plt.show()


# In[10]:


def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)


# In[11]:


# Third model
class QE_Dataset(Dataset):
    # 用於配對爭點與要件
    def __init__(self, dataframe, transformer):
        assert type(dataframe) == pd.core.frame.DataFrame, "Wrong input"
        self.df = dataframe
        self.questions = self.df['Issue']
        self.elements = self.df['Element']
        self.embedding = transformer
        self.len = len(self.df)

    def __getitem__(self, idx):
        question = self.questions[idx]
        element = self.elements[idx]
        label = torch.Tensor([label_to_numbers[element]])
        embedding = torch.from_numpy(self.embedding.encode(question))
        return embedding, label, question
    
    def __len__(self):
        return self.len


# In[12]:


class Third_model(nn.Module):
    def __init__(self, num_elements):
        super().__init__()
        self.embedding = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.fc1 = nn.Linear(in_features=768, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=num_elements)       

    def forward(self, t):
        # (1) Input layer
        t = t
        # (2) hidden linear layer
        t = self.fc1(t)
        t = F.relu(t)
        # (3) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        # (4) output layer
        t = self.out(t)
        t = F.relu(t)
        return t


# In[13]:


def train_model(model, loss_function, optimizer, num_epochs,device,trainloader):
    model.to(device)
    optimizer = optimizer(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
    # Run the training loop for defined number of epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 15)
        train_loss = 0
        train_correct = 0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            embeddings, labels, questions= data
            embeddings = embeddings.to(device)
            labels = labels.squeeze(1).long()
            labels = labels.to(device)
            preds = model(embeddings) #Pass batch
            loss = loss_function(preds, labels) #Calculate loss
            optimizer.zero_grad() #Clear gradient
            loss.backward() #Calculate gradient
            optimizer.step() #Update weights
            train_loss += loss.item()
            train_correct += get_num_correct(preds, labels)


# In[14]:


# Prediction funtions
def get_num_correct(preds, labels):
    return int(preds.argmax(dim=1).eq(labels).sum())

def Accuracy(correct_num, train_dataset):
    return round(float(correct_num / len(train_dataset))*100, 2)


# In[15]:


# The transformation between string labels and integer labels
label_to_numbers = {"傷害故意" : 0, "因果關係" : 1}
numbers_to_labels = {v: k for k, v in label_to_numbers.items()}


# In[16]:


QEdf = pd.read_csv("QandE.csv")
train_dataset = QE_Dataset(QEdf,Base_model)


# In[17]:


model=Third_model(2)
model.to(DEVICE)
trainloader = DataLoader(dataset = train_dataset,
         batch_size = BATCH_SIZE,
         shuffle=True)


# In[18]:


refresh_cuda_memory()


# In[19]:


if ToTrain:
    print("Use full training data to train the model.")
    # Train
    train_model(model, F.cross_entropy, optim.SGD, NUM_EPOCHS, DEVICE, trainloader)


# In[21]:


torch.cuda.empty_cache()
refresh_cuda_memory()


# In[22]:


with torch.no_grad():
    total_correct, total = 0, 0
    for i, data in enumerate(trainloader, 0):

        # Get inputs
        embeddings, labels, questions = data
        embeddings = embeddings.to(DEVICE)
        labels = labels.squeeze(1).long()
        labels = labels.to(DEVICE)

        # Generate outputs
        pred = model(embeddings)

        # Sum up correct
        total += labels.size(0)
      
        _, predicted = torch.max(pred.data, 1)
        QtoE = {}
        for i in predicted.detach().numpy():
            if QtoE.__contains__(numbers_to_labels[int(i)]) == False:                
                QtoE[numbers_to_labels[int(i)]] = []
            QtoE[numbers_to_labels[int(i)]].append(questions[int(i)])
        total_correct += (predicted == labels).sum().item()


# In[25]:


if __name__ == "__main__":
    # First model 處理後的結果
    print("原判決書:")
    
    for line in verdict:
        print(line)
    print("\n","-"*125,"\n")
    print("精簡後判決書:")
    
    for name, n_verdict in verdicts.items():
        for line in n_verdict:
            print(line)
    print("\n","-"*125,"\n")
    
    for evidence, sentence in sentence_to_evidence.items():
        print(f"{evidence} 對應到:")
        print(sentence,"\n")
    print("\n","-"*125,"\n")
    for element, question in QtoE.items():
        print(f"{element} 對應到:")
        print(question,"\n")
        


# In[ ]:




