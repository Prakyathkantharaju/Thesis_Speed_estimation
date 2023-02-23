import torch
from torch.utils.data import Dataset, DataLoader

class IMU(Dataset):
    def __init__(self,X, y, size = 4*148,num_features=9):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        self.n_samples = X.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(X.reshape(-1,num_features,size)).float() # size [n_samples, n_features]
        self.y_data = torch.from_numpy(y).float() # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples