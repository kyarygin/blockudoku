import random

from dataclasses import dataclass, field
from itertools import product, permutations
from typing import Tuple, List, Union
from copy import deepcopy


BOARD_SIZE = 9
BLOCK_SIZE = 3

@dataclass
class Figure:
    shape: List[List[int]] = field(default_factory=list)
    size: List[int] = field(init=False)
    mass: int = field(init=False)

    def __post_init__(self):
        self.size = [len(self.shape), len(self.shape[0])]
        self.mass = sum(sum(row) for row in self.shape)

    def __getitem__(self, pos: Tuple[int, int]) -> int:
        i, j = pos
        return self.shape[i][j]

    def __hash__(self):
        return hash(tuple(tuple(tile for tile in row) for row in self.shape))

    def __str__(self) -> str:
        output = '\n'.join(
            ' '.join('X' if tile else '.' for tile in row)
            for row in self.shape
        )
        return output

@dataclass
class FigureSet:
    storage: List[Figure] = field(default_factory=list)
    n_figures: int = field(init=False)

    def __post_init__(self):
        self.n_figures = len(self.storage)

    def __getitem__(self, key: int) -> Figure:
        return self.storage[key]

    @staticmethod
    def load(path) -> 'FigureSet':
        with open(path, 'r') as f:
            raw_data = f.read().strip()
        figures = [
            Figure(
                shape=[
                    [
                        int(tile)
                        for tile in raw_row.split(' ')
                    ]
                    for raw_row in raw_figure.split('\n')
                ]
            )
            for raw_figure in raw_data.split('\n\n')
        ]
        return FigureSet(figures)

    def sample(self, n) -> List[Figure]:
        return random.choices(self.storage, k=n)

@dataclass
class Board:
    state: List[List[int]] = field(
        default_factory=lambda: [[0 for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
    )

    def __str__(self) -> str:
        output = '\n'.join(
            ' '.join('X' if tile else '.' for tile in row)
            for row in self.state
        )
        return output

    def __getitem__(self, pos: Tuple[int, int]) -> int:
        i, j = pos
        return self.state[i][j]

    def __setitem__(self, pos: Tuple[int, int], value: int):
        i, j = pos
        self.state[i][j] = value

    def does_figure_fits(self, figure: Figure, i: int, j: int) -> bool:
        if (BOARD_SIZE - figure.size[0] < i) or (BOARD_SIZE - figure.size[1] < j):
            return False
        for fi, fj in product(range(figure.size[0]), range(figure.size[1])):
            figure_tile = figure[fi, fj]
            board_tile = self[fi+i, fj+j]
            if figure_tile and board_tile:
                return False
        return True

    def add_figure(self,
                   figure: Figure,
                   i: int, j: int,
                   inplace=True) -> Union[int, Tuple['Board', int]]:
        board_to_change = self if inplace else Board(deepcopy(self.state))
        for fi, fj in product(range(figure.size[0]), range(figure.size[1])):
            board_to_change[fi+i, fj+j] += figure[fi, fj]
        score = figure.mass
        if inplace:
            return score
        else:
            return board_to_change, score

    def update(self) -> int:
        full_rows = [
            i for i in range(BOARD_SIZE)
            if all(self[i, j] for j in range(BOARD_SIZE))
        ]
        full_cols = [
            j for j in range(BOARD_SIZE)
            if all(self[i, j] for i in range(BOARD_SIZE))
        ]
        full_blocks = [
            (bi, bj) for (bi, bj) in product(range(BOARD_SIZE // BLOCK_SIZE),
                                             range(BOARD_SIZE // BLOCK_SIZE))
            if all(
                self[bi*BLOCK_SIZE+i, bj*BLOCK_SIZE+j]
                for i, j in product(range(BLOCK_SIZE), range(BLOCK_SIZE))
            )
        ]
        for i, j in product(full_rows, range(BOARD_SIZE)):
            self[i, j] = 0
        for i, j in product(range(BOARD_SIZE), full_cols):
            self[i, j] = 0
        for (bi, bj), i, j in product(full_blocks, range(BLOCK_SIZE), range(BLOCK_SIZE)):
            self[bi*BLOCK_SIZE+i, bj*BLOCK_SIZE+j] = 0
        multiplier = len(full_rows) + len(full_cols) + len(full_blocks)
        return multiplier

class Game(object):

    def __init__(self):
        self.figure_set = FigureSet.load('data/figures')
        self.current_board = Board()
        self.current_score = 0
        self.current_round = 0

        self.current_series_multiplier = 1

    def round(self, seed=0):
        three_figures = self.figure_set.sample(3)
        figures_orders = {p for p in permutations(three_figures)}
        output_boards = []

        # for order in figures_orders:
        #     for figure in order:
        #         for i, j in product(range(BOARD_SIZE), range(BOARD_SIZE)):
        #             if

    def update(self, new_board, upscore):
        pass


if __name__ == '__main__':
    g = Game()
    g.round()
