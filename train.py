
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, num_layers)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])


# TODO: Load the dataset
train_dataset = VisualOdometryDataset(
    dataset_path="./dataset/train",
    transform=transform,
    sequence_length=sequence_length,
    validation=False
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


# train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
losses = []

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels, _ in tqdm(train_loader, f"Epoch {epoch + 1}:"):

        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        
        # Calculating the loss function
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Storing the losses in a list for plotting
        running_loss+=loss
    running_loss/=len(train_loader)

    losses.append(running_loss.cpu().detach())

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss}")

torch.save(model.state_dict(), "./vo.pt")
