import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, inputSize, hidden1Size, hidden2Size, numClasses):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(inputSize, hidden1Size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden1Size, hidden2Size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden2Size, numClasses)
        self.logsm1 = nn.LogSoftmax(dim=1)

    def forward(self, data):
        out = self.layer1(data)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        out = self.logsm1(out)
        return out