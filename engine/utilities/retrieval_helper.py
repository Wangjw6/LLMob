import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

def calculate_relative_features(date1, date2):
    delta = date1 - date2
    return [
        delta.days/365.,
        int(delta.days % 7 == 0),
        date1.month == date2.month,
    ]


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

def map_traj2mat(traj, class_loc_map, act_map, interval=60):
    mat = np.zeros((int(1440 / interval), len(act_map)))
    traj = traj.replace(":00, ", " at ")
    traj_ = traj.split(" at ")
    i = 0
    while i < len(traj_) - 1:
        loc = traj_[i].split("#")[0]
        time_str = traj_[i + 1]
        act_class = class_loc_map[loc]
        time_obj = None
        if '.' not in time_str:
            time_obj = datetime.strptime(time_str, '%H:%M')
        else:
            time_obj = datetime.strptime(time_str, '%H:%M:%S.')
        if time_obj is None:
            continue
        minutes = time_obj.hour * 60 + time_obj.minute
        mat[int(minutes / interval), act_map[act_class]] += 1
        i += 2
    return mat


def act_mat_compute(traj1, traj2, class_loc_map):
    act_map = {v: id_ for id_, v in enumerate(list(set(class_loc_map.values())))}
    mat1 = map_traj2mat(traj1, class_loc_map, act_map)
    mat2 = map_traj2mat(traj2, class_loc_map, act_map)
    comp = np.where((mat1 == mat2) & (mat1 >= 1))[0].shape[0] / max(np.where((mat1 >= 1))[0].shape[0],
                                                                    np.where((mat2 >= 1))[0].shape[0])
    return comp


class NodeWithScore:
    def __init__(self, node, score, input_):
        self.node = node
        self.score = score
        self.input_ = input_


class DeepModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TemporalRetriever:
    def __init__(self, nodes, similarity_top_k, is_train=None, class_id_map=None):
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        self.feature_size = 3
        if is_train is not None:
            self.calibrate_dataset = ContrastiveDataset(nodes,
                                                        eval=act_mat_compute,
                                                        num_pairs=3,
                                                        class_id_map=class_id_map)
            self._model = DeepModel(self.feature_size, 64, 1)
            self.optimizer = optim.Adam(self._model.parameters(), lr=0.002)
            self.criterion = torch.nn.MSELoss()
            self.calibrate_score_function(num_epochs=1500)

    def retrieve(self, query_str):
        scored_nodes = self.get_scored_nodes(query_str)
        nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)
        retrieved_nodes = [node.node for node in nodes[: self.similarity_top_k]]
        return retrieved_nodes

    def get_scored_nodes(self, query_date):
        _nodes = []
        query_date = datetime.strptime(query_date, "%Y-%m-%d")
        for node in self.nodes:
            node_date = datetime.strptime(node.split(": ")[0].split(" ")[-1], "%Y-%m-%d")
            score = self.get_similarity_score(query_date, node_date)
            _nodes.append(NodeWithScore(node=node, score=score, input_=input_from_date(query_date, node_date)))
        return _nodes

    def get_similarity_score(self, query_date, node_date):
        input_ = input_from_date(query_date, node_date)
        score = self._model(torch.tensor(input_).float())
        return score

    def calibrate_score_function(self, batch_size=64, num_epochs=10):
        dataloader = DataLoader(self.calibrate_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for batch in dataloader:
                positive_pairs = batch[:, 0, :].reshape(-1, self.feature_size)
                positive_scores = self._model(torch.tensor(positive_pairs).float())
                negative_pairs = batch[:, 1:, :].reshape(-1, self.calibrate_dataset.num_pairs - 1, self.feature_size)
                negative_scores = self._model(torch.tensor(negative_pairs).float()).reshape(batch.shape[0], -1)
                logits = torch.cat([positive_scores, negative_scores], dim=1)
                loss = -torch.log_softmax(logits, dim=1)[:, 0]
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print("Calibration finished!")


def retrieve_loc(person, route):
    area = []
    loc_in_retrieve = route.split(": ")[1].replace(",", " at ").split(" at ")[::2]
    selected_loc_cat = {}
    for loc in loc_in_retrieve:
        loc = loc.lstrip().rstrip()
        c = person.loc_cat[loc.split("#")[0]]
        for k, v in person.area_freq.items():
            k = k.replace(".", "")
            if person.loc_cat[k.split("#")[0]] == c:
                if c not in selected_loc_cat:
                    selected_loc_cat[c] = 1
                else:
                    selected_loc_cat[c] += 1
                if selected_loc_cat[c] <= 7:
                    area.append(k)
                else:
                    break
    return area
