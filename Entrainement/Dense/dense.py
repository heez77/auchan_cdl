import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseModel(nn.Module):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.dense1 = nn.Linear(158, 256)  # equivalent to Dense in keras
        self.dense2 = nn.Linear(256,128)
        self.dense3 = nn.Linear(128, 79)
<<<<<<< Updated upstream

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x=F.relu(self.dense2(x))
=======
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
>>>>>>> Stashed changes
        x = F.softmax(self.dense3(x), dim=0)
        return x

model = DenseModel()

<<<<<<< Updated upstream
# model = nn.Sequential(
#     nn.Linear(158, 256),
#     nn.Linear(256,79)
# )

=======
>>>>>>> Stashed changes
import pandas as pd
# Preprocessing
df_label = pd.read_csv("C:\\Users\\Tijoxa\\Documents\\GitHub\\auchan_cdl\\labels_en_fr.csv")
labels = df_label.fr.tolist()
for i,l in enumerate(labels):
    if l.endswith(' '):
        labels[i] = l[:-1]
dico = {label:i for i, label in enumerate(labels)}
df = pd.read_csv('C:\\Users\\Tijoxa\\Documents\\GitHub\\auchan_cdl\\dense_train.csv', index_col=False)
del df['image']
del df['description']
del df['Unnamed: 0']
y_train = df['label'].tolist()
del df['label']
X_train = df.to_numpy()
X_train = torch.tensor(X_train)

df2 = pd.read_csv('C:\\Users\\Tijoxa\\Documents\\GitHub\\auchan_cdl\\dense_train.csv', index_col=False)
del df2['image']
del df2['description']
del df2['Unnamed: 0']
y_val = df2['label'].tolist()
del df2['label']
X_val = df2.to_numpy()
X_val = torch.tensor(X_val)

import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float32')[y]

y_train = [dico[t] for t in y_train]
y_train = to_categorical(y_train, num_classes=79)
y_train = torch.tensor(y_train)

y_val = [dico[t] for t in y_val]
y_val = to_categorical(y_val, num_classes=79)
y_val = torch.tensor(y_val)

x = X_train
y = y_train

# train
<<<<<<< Updated upstream
criterion = torch.nn.MSELoss(reduction='sum')
=======
criterion = torch.nn.MSELoss(reduction='sum')  # changer
>>>>>>> Stashed changes
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5000):
    y_pred = model(x.float())
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

df3 = pd.read_csv('C:\\Users\\Tijoxa\\Documents\\GitHub\\auchan_cdl\\dense_test.csv', index_col=False)
del df3['image']
del df3['description']
del df3['Unnamed: 0']
y_test = df3['label'].tolist()
del df3['label']
X_test = df3.to_numpy()
X_test = torch.tensor(X_test)

predictions = model(X_test.float())
predictions = [list(p) for p in  predictions]

df_test =  pd.read_csv('C:\\Users\\Tijoxa\\Documents\\GitHub\\auchan_cdl\\dense_test.csv', index_col=False)

vrai = 0
tot = len(df_test)
for i in range(len(df_test)):
    if df_test.label.iloc[i] == labels[predictions[i].index(max(predictions[i]))]:
        vrai +=1
print("vrai/tot = ")
print(vrai/tot)