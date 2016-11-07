import pandas as pd
import numpy as np
import os
from datetime import time


class DataLoader():
    """Create a new dataframe of raw data """
    def __init__(self, data_dir, batch_size, num_steps):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_steps = num_steps

        file = os.path.join(data_dir, "stkhf2013_000026sz.csv")
        self.preprocess(file)
        self.create_batches()
        self.reset_batch_pointer()


    def preprocess(self, file):
        reader = pd.read_csv(file, encoding='gbk', iterator=True)
        df = reader.get_chunk(200000)
        df.index = pd.DatetimeIndex(df['QTime'])
        pct2 = df['ChangePCT1'].between_time(time(9,30), time(15,0))
        pct = pct2.reset_index()['ChangePCT1']
        self.pct = np.array(pct)

    def create_batches(self):
        self.num_batches = int(self.pct.size / (self.batch_size * self.num_steps))
        self.pct = self.pct[:self.num_batches * self.batch_size * self.num_steps]
        xdata = self.pct
        ydata = np.copy(self.pct)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


    def reset_batch_pointer(self):
        self.pointer = 0










