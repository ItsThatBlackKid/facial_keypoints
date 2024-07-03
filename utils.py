import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn

def load_images():
    data_np = np.load('data/face_images.npz')
    images = np.moveaxis(data_np['face_images'], -1,0)
    return images

def load_csv():
    return pd.read_csv('data/facial_keypoints.csv')

def fill_na(df):
    for i in df.columns[df.isnull().any(axis=0)]:     
        df[i].fillna(df[i].mean(),inplace=True)

def plot_landmark(idx, images, points_df):
    plt.imshow(images[idx], cmap="gray")
    plt.scatter(points_df.iloc[idx][0: -1: 2], points_df.iloc[idx][1: : 2], cmap='y')
    plt.show()

def plot_landmark_single(img, land_marks):
    print(land_marks)
    plt.imshow(img, cmap="gray")
    plt.scatter(land_marks[0: -1: 2], land_marks[1: : 2], cmap='cool')
    plt.show()
    

class FaceLandmarkDataset(Dataset):
    def __init__(self, device='cuda', transform=None):
        self.device = device
        self.landmarks_frame = load_csv()
        fill_na(self.landmarks_frame)
        images = load_images()
        self.images =torch.from_numpy(images).type(torch.float)
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.images[idx]
        landmarks = torch.from_numpy(self.landmarks_frame.iloc[idx].to_numpy()).type(torch.float64)
        sample = {'image': img, 'landmarks': landmarks}
        
        return sample
    
    
def training_loop(model, optimizer, loss_fn=nn.L1Loss(), batch_size=32, epochs=40, train_test=split_dataset())

