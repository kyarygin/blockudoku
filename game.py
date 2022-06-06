import random

from dataclasses import dataclass, field
from itertools import product, permutations
from typing import Tuple, List, Set, Union, Optional
from copy import deepcopy

from time import time


BOARD_SIZE = 9
BLOCK_SIZE = 3

@dataclass
class Figure:
    shape: Tuple[Tuple[int]] = field(default_factory=tuple)
    size: Tuple[int] = field(init=False)
    mass: int = field(init=False)

    def __post_init__(self):
        self.size = (len(self.shape), len(self.shape[0]))
        self.mass = sum(sum(row) for row in self.shape)

    def __getitem__(self, pos: Tuple[int, int]) -> int:
        i, j = pos
        return self.shape[i][j]

    def __hash__(self):
        # return hash(tuple(tuple(tile for tile in row) for row in self.shape))
        return hash(self.shape)

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
                shape=tuple(
                    tuple(
                        int(raw_tile)
                        for raw_tile in raw_row.split(' ')
                    )
                    for raw_row in raw_figure.split('\n')
                )
            )
            for raw_figure in raw_data.split('\n\n')
        ]
        return FigureSet(figures)

    def sample(self, n) -> List[Figure]:
        return random.choices(self.storage, k=n)

@dataclass
class BoardState:
    state: Tuple[Tuple[int]] = field(
        default_factory=lambda: tuple(tuple(0 for j in range(BOARD_SIZE)) for i in range(BOARD_SIZE))
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

    def possible_figure_places(self, figure: Figure) -> List[Tuple[int, int]]:
        possible_places = [
            (i, j)
            for (i, j) in product(range(BOARD_SIZE), range(BOARD_SIZE))
            if self.does_figure_fits(figure, i, j)
        ]
        return possible_places

    def add_figure(self, figure: Figure, fi: int, fj: int) -> 'BoardState':
        new_state = tuple(
            tuple(
                self[i, j] + figure[i-fi, j-fj]
                if (fi <= i < fi+figure.size[0]) and (fj <= j < fj+figure.size[1])
                else self[i, j]
                for j in range(BOARD_SIZE)
            )
            for i in range(BOARD_SIZE)
        )
        return BoardState(new_state)

    def update(self) -> 'BoardState':
        full_rows = {
            i for i in range(BOARD_SIZE)
            if all(self[i, j] for j in range(BOARD_SIZE))
        }
        full_cols = {
            j for j in range(BOARD_SIZE)
            if all(self[i, j] for i in range(BOARD_SIZE))
        }
        full_blocks = {
            (bi, bj) for (bi, bj) in product(range(BOARD_SIZE // BLOCK_SIZE),
                                             range(BOARD_SIZE // BLOCK_SIZE))
            if all(
                self[bi*BLOCK_SIZE+i, bj*BLOCK_SIZE+j]
                for i, j in product(range(BLOCK_SIZE), range(BLOCK_SIZE))
            )
        }
        new_state = tuple(
            tuple(
                0
                if (i in full_rows) or (j in full_cols) or ((i // 3, j // 3) in full_blocks)
                else self[i, j]
                for j in range(BOARD_SIZE)
            )
            for i in range(BOARD_SIZE)
        )
        # multiplier = len(full_rows) + len(full_cols) + len(full_blocks)
        return BoardState(new_state)


class Game(object):

    def __init__(self):
        self.figure_set = FigureSet.load('data/figures')
        self.current_board_state = BoardState()
        self.current_score = 0
        self.current_round = 0

        self.current_series_multiplier = 1

    def get_new_figures(self, n: int=3, seed: int=0) -> List[Figure]:
        random.seed(seed)
        return self.figure_set.sample(n)

    def possible_outcomes(self, three_figures: List[Figure]):
        figures_orders = {p for p in permutations(three_figures)}

        possibilities = []
        for order in figures_orders:
            current_order_possibilities = [(self.current_board_state, [])]
            for figure in order:
                t1 = time()
                current_order_possibilities = [
                    (
                        prev_board.add_figure(figure, i, j),
                        prev_history + [(figure, i, j)]
                    )
                    for prev_board, prev_history in current_order_possibilities
                    for i, j in prev_board.possible_figure_places(figure)
                ]
                t2 = time()
                print(f'{(t2 - t1):.3f} sec')
            possibilities += current_order_possibilities

        return possibilities


    def update(self, new_board, upscore):
        pass


if __name__ == '__main__':
    g = Game()
    nf = g.get_new_figures(3, seed=1)
    for f in nf:
        print(f)
        print()
    po = g.possible_outcomes(nf)
    print(len(po))
    po_1_board = po[5000][0]
    print(po_1_board)

