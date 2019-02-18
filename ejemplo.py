import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import model
import funciones
# modulos pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
# from torch.autograd import Variable


numeroMuestras = 50000
numeroEpoch = 50
batchSize = 1000

X = np.empty([numeroMuestras, 3])
X[:, 0] = np.random.rand(numeroMuestras)
X[:, 1] = np.random.rand(numeroMuestras)
for i in range(numeroMuestras):
    X[i, 2] = funciones.definir_salida(X[i, 0], X[i, 1])

train, test = train_test_split(X, test_size=0.3)


trainDataset = funciones.MiDataset(train)
testDataset = funciones.MiDataset(test)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=testDataset.__len__(), shuffle=False)

classWeight = torch.from_numpy(funciones.calcula_class_weights(train[:, -1])).float()

miRed = model.DNN(2, 300, 150, 5)
lossFunction = nn.NLLLoss(weight=classWeight)
optimizer = Adam(miRed.parameters())

lossTrain = []
lossTest = []
minAccuracy = 0

for epoch in range(numeroEpoch):
    for data, target in trainLoader:
        data = data.detach().requires_grad_(True).float()
        target = target.detach().requires_grad_(True).long()
        optimizer.zero_grad()
        out = miRed(data)
        loss = lossFunction(out, target)
        loss.backward()
        optimizer.step()

        _, salida = torch.max(out, 1)

    lossTrain = np.append(lossTrain, loss.item())
    print("Epoch {} de {}\tLoss:{}\tf1_score:{}".format(epoch,
                                                        numeroEpoch,
                                                        loss.item(),
                                                        f1_score(target.detach().numpy().astype(int),
                                                                 salida.detach().numpy().astype(int),
                                                                 average='micro')))
    if epoch % 5 == 0:
        for dataTest, targetTest in testLoader:
            dataTest = dataTest.detach().requires_grad_(True).float()
            targetTest = targetTest.detach().requires_grad_(True).long()
            miRed.eval()
            out = miRed(dataTest)
            loss = lossFunction(out, targetTest)
            miRed.train()
            _, salidas = torch.max(out, 1)
            accuracy = f1_score(targetTest.detach().numpy().astype(int),
                                salidas.detach().numpy().astype(int),
                                average='micro')
        print("Evaluación: {}".format(accuracy))
        lossTest = np.append(lossTest, loss.item())

        if accuracy > minAccuracy:
            torch.save(miRed, "./Red.pt")
            print("Modelo guardado")

plt.plot(range(numeroEpoch), lossTrain, 'r')
plt.plot(range(0, numeroEpoch, 5), lossTest, 'g')
plt.show()

