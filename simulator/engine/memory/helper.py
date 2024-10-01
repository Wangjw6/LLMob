import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import random


def extract_date_features(date):
    return [
        date.year, date.month, date.day,
        date.weekday(), date.timetuple().tm_yday
    ]


def calculate_relative_features(date1, date2):
    delta = date1 - date2
    return [
        delta.days/365.,
        int(delta.days % 7 == 0),
        date1.month == date2.month,
    ]


def normalize_features(features):
    # Dummy function - in practice, you'll need to scale based on your dataset
    return [feature / 100.0 for feature in features]


def encode_dates(date1, date2):
    relative_features = calculate_relative_features(date1, date2)

    return np.array(relative_features)

def input_from_date(query_date, node_date):
    f = encode_dates(query_date, node_date)
    return f


# Define a custom dataset class
class ContrastiveDataset(Dataset):
    def __init__(self, data, eval=None, num_pairs=10, class_id_map=None):
        self.data = data
        self.eval = eval
        self.num_pairs = num_pairs
        self.eval_pairs = []
        self.class_id_map = class_id_map

        self._create_pairs()


    def _create_pairs(self):
        scores = np.zeros((len(self.data), len(self.data))) - 100.
        node_dates = [datetime.strptime(s.split(": ")[0].split(" ")[-1], "%Y-%m-%d") for s in self.data]
        activities = [s.split(": ")[1] for s in self.data]
        l = len(self.data)
        # l = self.num_pairs
        for i in range(l):
            sample = []
            for j in range(l):
                if i != j and scores[i][j] < 0.:
                    score = self.eval(activities[i], activities[j], self.class_id_map)
                    scores[i][j] = score
                    scores[j][i] = score

            sorted_indices = sorted(range(len(self.data)), key=lambda k: scores[k].reshape(-1).tolist(), reverse=True)
            query_date = node_dates[i]
            node_date = node_dates[sorted_indices[0]] # most similar / positive sample
            input_ = input_from_date(query_date, node_date)
            sample.append(input_)

            for k in range(self.num_pairs-1):
                node_date = node_dates[sorted_indices[-k+1]] # least similar / negative sample
                input_ = input_from_date(query_date, node_date)
                sample.append(input_)
            sample = np.array(sample).reshape(self.num_pairs, -1)
            self.eval_pairs.append(sample)

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, idx):
        return self.eval_pairs[idx]
