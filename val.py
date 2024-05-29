
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import *


# Create the visual odometry model
model = VisualOdometryModel(hidden_size=hidden_size, num_layers=num_layers, \
                            bidirectional=bidirectional, lstm_dropout=lstm_dropout)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])


# Load dataset
val_dataset = VisualOdometryDataset(
    dataset_path="./dataset/val", #/rgbd_dataset_freiburg3_walking_rpy_validation",
    transform=transform,
    sequence_length=sequence_length,
    validation=True
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# val
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("vo.pt")
model.to(device)
model.eval()

validation_string = ""
position = torch.tensor([0.0] * 7)

with torch.no_grad():
    for images, labels, timestamp in tqdm(val_loader, f"Validating:"):

        images = images.to(device)
        labels = labels.to(device)
        timestamp = timestamp.numpy().tolist()

        target = model(images).cpu()
        
        #For in batch
        for idx in range(len(timestamp)):
            position += target[idx]

            #Text to write in txt
            validation_string+=str(timestamp[idx])
            for pose in position.numpy().tolist():
                validation_string+= f",{pose}"
            validation_string+="\n"
        
# Sve text in txt
f = open("validation.txt", "a")
f.write(validation_string)
f.close()
