dev = False if __file__ == '/tmp/Answer.py' else True

import sys
import numpy as np
from scipy.signal import convolve2d


def debug(message) -> None:
    print(message, file=sys.stderr, flush=True)


def input_stream(filename):
    print_logs = [line.strip() for line in open(f"./input_logs/{filename}", 'r')]
    idx = -1

    def inner():
        nonlocal idx
        idx += 1
        if idx < len(print_logs):
            return print_logs[idx]
        else:
            raise EOFError("End of input stream")
    return inner


class Board:

    def __init__(self, input_array):
        self._input_array = input_array
        self.manhattan_kernel = np.array([[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0]])

    @property
    def scrap_amount(self):
        return self._input_array[:, :, 0]

    @property
    def owner(self):
        return self._input_array[:, :, 1]

    @property
    def units(self):
        return self._input_array[:, :, 2]

    @property
    def recycler(self):
        return self._input_array[:, :, 3]

    @property
    def can_build(self):
        return self._input_array[:, :, 4]

    @property
    def can_spawn(self):
        return self._input_array[:, :, 5]

    @property
    def in_range_of_recycler(self):
        return self._input_array[:, :, 6]

    @property
    def is_grass(self):
        return self.scrap_amount == 0

    @property
    def scrap_amount_k3_manh(self):
        return convolve2d(self.scrap_amount, self.manhattan_kernel, mode='same').astype(int)

    @property
    def owner_me(self):
        return self.owner == 1

    @property
    def owner_me_k3_manh(self):
        return convolve2d(self.owner_me, self.manhattan_kernel, mode='same').astype(int)

    @property
    def owner_op(self):
        return self.owner == 0

    @property
    def owner_op_k3_manh(self):
        return convolve2d(self.owner_op, self.manhattan_kernel, mode='same').astype(int)

    @property
    def owner_ne(self):
        return self.owner == -1

    @property
    def owner_ne_k3_manh(self):
        return convolve2d(self.owner_ne, self.manhattan_kernel, mode='same').astype(int)

    @property
    def tile_live(self):
        return (self.scrap_amount > 0) * (self.recycler == 0)

    @property
    def tile_live_k3_manh(self):
        return convolve2d(self.tile_live, self.manhattan_kernel, mode='same').astype(int)

    @property
    def tile_dead(self):
        return self.tile_live == 0

    @property
    def tile_dead_k3_manh(self):
        return convolve2d(self.tile_dead, self.manhattan_kernel, mode='same').astype(int)

    @property
    def units_op(self):
        return self.units * (self.owner == 0)

    @property
    def units_op_k3_manh(self):
        return convolve2d(self.units_op, self.manhattan_kernel, mode='same').astype(int)

    @property
    def units_me(self):
        return self.units * (self.owner == 1)

    @property
    def units_me_k3_manh(self):
        return convolve2d(self.units_me, self.manhattan_kernel, mode='same').astype(int)



class InputOutput:

    def __init__(self, filename: str = "unittest.txt"):
        self.input = input_stream(filename)

    def get_input(self, n: int = 1) -> None:
        lst = [self.input().split() for _ in range(n)]
        return [int(item) for sublist in lst for item in sublist]

    def send_action(self, actions: list):
        message = ";".join(action.__str__() for action in actions)
        print(f"{message}")


def main():
    io = InputOutput('right.txt')
    col, row = io.get_input()

    while True:
        try:
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
            board = Board(input_array)
            io.send_action(['WAIT'])
        except EOFError:
            break

if __name__ == '__main__':
    main()
