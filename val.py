
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


# TODO: Load dataset
val_dataset = VisualOdometryDataset(
    dataset_path="./dataset/val",
    transform=transform,
    sequence_length=sequence_length,
    validation=True
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)


# val
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.load_state_dict(torch.load("vo.pt"))
model.eval()

validation_string = ""
position = torch.tensor([0.0] * 7)

with torch.no_grad():
    for images, labels, timestamp in tqdm(val_loader, f"Validating:"):

        images = images.to(device)
        labels = labels.to(device)

        target = model(images).cpu().numpy().tolist()[0]
        position += target

        # TODO: add the results into the validation_string
        validation_string+=str(timestamp)
        for pose in position:
            validation_string+= f",{pose}"
        validation_string+="\n"
        


f = open("validation.txt", "a")
f.write(validation_string)
f.close()
