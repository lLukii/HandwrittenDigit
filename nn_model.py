import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,),(1,))
])
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
input_size = 784
hidden_size = 512
output_size = 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_forward(x)
        return output
    
    def xavierInit(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

model = NeuralNetwork()
model.xavierInit()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 12


loss_cache = []
def train_loop(dataloader):
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        # forward prop
        pred = model((X > 0).to(torch.float32))
        loss = loss_func(pred, Y)
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_cache.append(loss.item())
        if batch % 1000 == 0:  
            print(f"Current loss: {loss.item()}")

def test_loop(dataloader):
    model.eval() # sets to evaluation mode
    size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for X, Y in dataloader: 
            pred = model((X > 0).to(torch.float32))
            correct += (pred.argmax(1) == Y).sum().item()

    print(f"Accuracy: {correct / size * 100}%")

def train_model():
    for t in range(epochs):
        print(f"Epoch #{t+1}")
        train_loop(train_loader)
        test_loop(test_loader)

def test_on_handwriting(data):
    model.eval() 
    data = torch.Tensor(data).reshape(1, 28, 28).to(torch.float32)
    with torch.no_grad():
        pred = model(data)
    print(pred)
    return int(pred.argmax(1))
