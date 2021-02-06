from torch.utils.data import Dataset


class GenericDataset(Dataset):
    
    """Use this when each sample of the dataset contains a sequence
    of video frames.
    """

    def __init__(self, X, forecast_horizon, lags):
        self.X = X
        self.forecast_horizon = forecast_horizon
        self.lags = lags

    def __len__(self):
        length = len(self.X)
        return length

    def __getitem__(self, idx):
        x = self.X[idx, :self.lags]
        y = self.X[idx, self.lags:self.lags+self.forecast_horizon]

        return idx, x, y

class SpatiotemporalDataset(Dataset):
    
    """Use this with an array of sequential images. ie X is an
    array of sequential video frames.
    """

    def __init__(self, X, forecast_horizon, lags):
        self.X = X
        self.forecast_horizon = forecast_horizon
        self.lags = lags

    def __len__(self):
        length = len(self.X) - self.forecast_horizon - self.lags + 1
        return length

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.lags]
        y = self.X[idx+self.lags:idx+self.lags+self.forecast_horizon]

        return idx, x, y