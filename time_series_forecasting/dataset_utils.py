from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    
    """Use this when each sample of the dataset contains a contiguous
    time series.
    """

    def __init__(self, X, forecast_horizon, lags, aux_series=None):
        self.X = X
        self.forecast_horizon = forecast_horizon
        self.lags = lags
        self.aux_series = aux_series

    def __len__(self):
        length = len(self.X) - self.lags - self.forecast_horizon + 1
        return length

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.lags]
        y = self.X[idx+self.lags:idx+self.lags+self.forecast_horizon]
        
        if self.aux_series is not None:
            aux = self.aux_series[idx:idx+self.lags]
            x = np.concatenate([x, aux], axis=1)

        return idx, x, y