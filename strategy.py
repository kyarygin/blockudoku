from abc import ABC, abstractmethod

from typing import Type
from state_value_network import StateValueNetwork

import numpy as np
from tqdm import tqdm
from math import floor
import torch
import random

CHUNK_SIZE = 4096


class Strategy(ABC):
    @abstractmethod
    def best_state(self, states: np.array) -> int:
        pass

    @abstractmethod
    def top_n_states(self, states: np.array) -> np.array:
        pass

class RandomStrategy(Strategy):
    def __init__(self):
        super(Strategy, self).__init__()

    def best_state(self, states: np.array) -> int:
        return random.choice(range(states.shape[0]))

    def top_n_states(self, states: np.array, n: int) -> np.array:
        return np.array(random.choice(range(states.shape[0]), n))

class NNStrategy(Strategy):
    def __init__(self, model_class: Type[torch.nn.Module], model_path: str):
        super(Strategy, self).__init__()
        self.model = model_class()
        self.model.load_state_dict(torch.load(model_path))

    def best_state(self, states: np.array) -> int:
        states_t = torch.from_numpy(states).type(torch.float).unsqueeze(dim=1)
        max_value = -1
        max_index = 0
        for i_chunk, state_chunk in enumerate(states_t.split(CHUNK_SIZE)):
            preds = self.model(state_chunk).flatten()
            if torch.max(preds) > max_value:
                max_value = torch.max(preds)
                max_index = torch.argmax(preds) + CHUNK_SIZE*i_chunk
        return max_index

    def top_n_states(self, states: np.array, n: int) -> np.array:
        pass
        # states_t = torch.from_numpy(states).type(torch.float).unsqueeze(dim=1)
        # preds = self.model(states_t).flatten()
        # return torch.topk(preds, n).indices


random_strategy = RandomStrategy()
nn_strategy = NNStrategy(StateValueNetwork, 'models/de_novo.valid_0.227.model')
