import torch
import torch.nn as nn
from simulator.engine.memory.helper import *
from simulator.engine.evaluation.metrics import *
import torch.optim as optim


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
                negative_scores = self._model(torch.tensor(negative_pairs).float()).reshape(batch.shape[0],
                                                                                            -1)

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
