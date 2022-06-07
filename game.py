import random

from dataclasses import dataclass, field
from itertools import product, permutations
from typing import Tuple, List, Set, Union, Optional
from copy import deepcopy
from numba import njit
from time import time
from tqdm import tqdm

import numpy as np
from collections import Counter

BOARD_SIZE = 9
BLOCK_SIZE = 3


@njit(cache=True)
def np_all_axis0(x):
    out = np.ones(x.shape[1], dtype=np.bool8)
    for i in range(x.shape[0]):
        out = np.logical_and(out, x[i, :])
    return out

@njit(cache=True)
def np_all_axis1(x):
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

@njit
def does_figure_fits(board: np.array, figure: np.array, i: int, j: int) -> bool:
    if (BOARD_SIZE - figure.shape[0] < i) or (BOARD_SIZE - figure.shape[1] < j):
        return False
    board_slice = board[i:i+figure.shape[0], j:j+figure.shape[1]]
    return np.all(board_slice + figure <= 1)

@njit
def possible_figure_places(board: np.array, figure: np.array) -> List[Tuple[int, int]]:
    possible_places = [
        (i, j)
        for i in range(BOARD_SIZE)
        for j in range(BOARD_SIZE)
        if does_figure_fits(board, figure, i, j)
    ]
    return possible_places

@njit
def add_figure(board: np.array, figure: np.array, i: int, j: int) -> np.array:
    new_board = board.copy()
    new_board[i:i+figure.shape[0], j:j+figure.shape[1]] += figure
    return new_board

@njit
def update(board: np.array) -> np.array:
    full_cols = np_all_axis0(board)
    full_rows = np_all_axis1(board)
    full_blocks = [
        (bi, bj)
        for bi in range(BOARD_SIZE // BLOCK_SIZE)
        for bj in range(BOARD_SIZE // BLOCK_SIZE)
        if np.all(board[
            bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
            bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE
        ])
    ]
    new_board = board.copy()
    new_board[:, full_cols] = 0
    new_board[full_rows, :] = 0
    for (bi, bj) in full_blocks:
        new_board[
            bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
            bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE
        ] = 0
    return new_board

@njit
def possible_outcomes(start_board: np.array, ordered_figures: List[np.array]) -> np.array:
    possibilities = [start_board]
    for next_figure in ordered_figures:
        possibilities = [
            update(add_figure(board, next_figure, i, j))
            for board in possibilities
            for i, j in possible_figure_places(board, next_figure)
        ]
    return possibilities


class Game(object):

    def __init__(self):
        self.figure_set = self.load_figure_set('game_data/figures')
        self.current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)

    def load_figure_set(self, path: str):
        with open(path, 'r') as f:
            data = f.read().strip()
        figures = [
            np.array([
                [int(tile_str) for tile_str in row_str.split(' ')]
                for row_str in figure_str.split('\n')
            ])
            for figure_str in data.split('\n\n')
        ]
        return figures

    def reset(self):
        self.current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)

    def get_new_figures(self, k: int=3) -> List[np.array]:
        return random.choices(self.figure_set, k=k)

    def round_results(self, new_figures):
        round_outcomes = [
            outcome
            for order in permutations(new_figures)
            for outcome in possible_outcomes(self.current_board, order)
        ]
        return round_outcomes

    def random_round(self):
        new_figures = self.get_new_figures(3)
        for figure in new_figures:
            ij_list = possible_figure_places(self.current_board, figure)
            if not ij_list:
                return False
            i, j = random.choice(ij_list)
            self.current_board = add_figure(self.current_board, figure, i, j)
        return True

    def random_game(self):
        self.reset()
        game_states = []
        for i_round in range(1000):
            round_result = self.random_round()
            game_states.append(self.current_board)
            if not round_result:
                break
        return np.stack(game_states), np.arange(i_round, -1, -1)


if __name__ == '__main__':
    g = Game()
    game_states_list = []
    target_list = []
    for i in tqdm(range(100_000)):
        game_states, target = g.random_game()
        game_states_list.append(game_states)
        target_list.append(target)
    X = np.concatenate(game_states_list)
    y = np.concatenate(target_list)
    print(X.shape)
    print(y.shape)
    with open('train_data/X.npy', 'wb') as f:
        np.save(f, X)
    with open('train_data/y.npy', 'wb') as f:
        np.save(f, y)
