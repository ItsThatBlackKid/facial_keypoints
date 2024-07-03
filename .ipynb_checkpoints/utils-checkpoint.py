import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    plt.imshow(images[idx])
    plt.scatter(points_df.iloc[idx][0: -1: 2], points_df.iloc[idx][1: : 2], cmap='y')
    plt.show()

