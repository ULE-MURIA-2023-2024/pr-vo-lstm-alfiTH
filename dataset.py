
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable
from PIL import Image
from tqdm import tqdm



class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        # Get all directories
        directories = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        # For in all folders
        for subdir in directories:
            print(f"Load {subdir}")

            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            # if is training interpolate the ground_truth, the other wise it is 0.0
            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = torch.tensor(
                    [item[1] for item in self.interpolate_ground_truth(
                        rgb_paths, ground_truth_data)])

            else:
                interpolated_ground_truth = torch.tensor([0.0] * len(rgb_paths))

            #rgb_paths = (timestamp, path), ground_truth_data = ([pose])

            for i in tqdm(range(len(rgb_paths)-sequence_length+1)):
                data = []
                for pair in range(sequence_length):
                    # print(f"\rIndex:{i}, pair:{pair}, pose:{interpolated_ground_truth[i+pair]}", end="")
                    data.append(rgb_paths[i+pair][1])
                ground = interpolated_ground_truth[i] - interpolated_ground_truth[i+pair]
                self.sequences.append([data, ground, rgb_paths[i+pair][0]])

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.TensorType:

        # Load sequence of images
        sequence_images = []
        ground_truth_pos = self.sequences[idx][1]
        timestampt = self.sequences[idx][2]

        for img_path in self.sequences[idx][0]:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            sequence_images.append(img)

        #print(f"image:{sequence_images}\n ground{ground_truth_pos}\n time{timestampt}")
    
        # TODO: return the next sequence
        return torch.stack(sequence_images, dim=0), ground_truth_pos, timestampt

    def read_images_paths(self, dataset_path: str) -> Tuple[float, str]:

        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> Tuple[float, Tuple[float]]:

        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:

            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: Tuple[float, str],
            ground_truth_data: Tuple[float, Tuple[float]]
    ) -> Tuple[float, Tuple[float]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:

            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
