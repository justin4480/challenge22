dev = False if __file__ == '/tmp/Answer.py' else True

import sys
import numpy as np
from scipy.signal import convolve2d
# import matplotlib.pyplot as plt
# import seaborn as sns

verbose = False

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


def get_filter(rows, cols):
    n = max(rows, cols) * 2 + 1
    p = 0.9
    a = np.concatenate([np.arange(n//2, 0, -1), [0.25], np.arange(1, n//2+1, 1)], axis=0)
    X, Y = np.meshgrid(a, a)
    filter = p ** (X+Y)
    filter[n//2 - 1: n//2 + 2, n//2] = 1
    filter[n//2, n//2 - 1: n//2 + 2] = 1
    return filter


class Board:

    def __init__(self, input_array):
        self._input_array = input_array
        self.manhattan_3_3_kernel = np.array([[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0]])
        self.manhattan_kernel = get_filter(input_array.shape[0], input_array.shape[1])
        

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
        return convolve2d(self.scrap_amount, self.manhattan_3_3_kernel, mode='same').astype(int)

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
        return convolve2d(self.owner_op, self.manhattan_kernel, mode='same')

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

    @property
    def recycler_me(self):
        return self._input_array[:, :, 3] * (self.owner == 1)


# def get_move_actions(board):
#     min_max = lambda x: (x - x.min()) / (x.max() - x.min())

#     results = []
#     actions = []

#     for row, col in np.argwhere(board.units_me):
#         a = np.full_like(board.scrap_amount, 0, dtype=np.float64)
#         current_unit = np.zeros_like(a)
#         current_unit[row, col] = 1
#         units_me_k3_manh_excl_current_unit = convolve2d(
#             board.units * (board.owner == 1) * (current_unit == 0),
#             board.manhattan_kernel, mode='same'
#         ).astype(int)

#         a += +1 * min_max(board.scrap_amount_k3_manh)
#         a += +4 * min_max(board.owner_op_k3_manh)
#         a += -4 * min_max(board.owner_me_k3_manh)
#         a += +2 * min_max(board.owner_ne_k3_manh)
#         a += +1 * min_max(board.tile_live_k3_manh)
#         a += -1 * min_max(board.tile_dead_k3_manh)
#         a += +1 * min_max(board.units_op_k3_manh)        
#         a += -2 * min_max(units_me_k3_manh_excl_current_unit)

#         legal_moves = board.tile_live * convolve2d(current_unit, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='same')
#         scores = a * np.where(legal_moves, 1, np.nan)
#         max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
#         actions.append(f"MOVE 1 {col} {row} {max_indices[1]} {max_indices[0]}")
#         results.append({'row_col': (row, col), 'max_indices': max_indices, 'max_value': max_value})
    # if verbose:
        # debug('get_move_actions scores')
#         debug(scores)
#         if False:
#             default_params = {'linecolor': 'black', 'linewidths': .5, 'square': True, 'cbar': False, 'annot': True, 'fmt': ".2f"}
#             sns.heatmap(**default_params, data=b).set_title(f"{max_value=} Action: MOVE 1 {col} {row} {max_indices[1]} {max_indices[0]}")
#             plt.show()

#     return actions


def get_move_actions(board, weights):
    results = []
    actions = []
    min_max = lambda x: (x - x.min()) / (x.max() - x.min())

    a = np.full_like(board.scrap_amount, 0, dtype=np.float64)

    for feature, weight in weights.items():
        a += weight * min_max(getattr(board, feature))

    for row, col in np.argwhere(board.units_me):
        current_unit = np.zeros_like(a)
        current_unit[row, col] = 1
        units_me_k3_manh_excl_current_unit = convolve2d(
            board.units * (board.owner == 1) * (current_unit == 0),
            board.manhattan_kernel, mode='same'
        ).astype(int)

        a += -2 * min_max(units_me_k3_manh_excl_current_unit)

        legal_moves = board.tile_live * convolve2d(current_unit, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='same')
        scores = a * np.where(legal_moves, 1, np.nan)
        if np.all(np.isnan(scores)):
            break
        max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
        actions.append(f"MOVE 1 {col} {row} {max_indices[1]} {max_indices[0]}")
        results.append({'row_col': (row, col), 'max_indices': max_indices, 'max_value': max_value})

    return actions


def get_build_random(board):
    options = np.argwhere(board.can_build)
    row, col = options[np.random.randint(0, len(options))]
    return [] if row is None else [f"BUILD {col} {row}"]


def get_spawn_random(board, k=1):
    options = np.argwhere(board.can_spawn)
    row, col = options[np.random.randint(0, len(options))]
    return [] if row is None else [f"SPAWN {k} {col} {row}"]

# def get_spawn_1(board, k=1):
#     min_max = lambda x: (x - x.min()) / (x.max() - x.min())

#     a = np.full_like(board.scrap_amount, 0, dtype=np.float64)

#     a += +1 * min_max(board.scrap_amount_k3_manh)
#     a += +9.0 * min_max(board.owner_op_k3_manh)
#     a += +3.0 * min_max(board.owner_op_k7_ones)
#     a += +1.0 * min_max(board.owner_op_k11_ones)
#     a += +0.3 * min_max(board.owner_op_k15_ones)
#     a += +0.1 * min_max(board.owner_op_k19_ones)
#     a += -5 * min_max(board.owner_me_k3_manh)
#     a += +5 * min_max(board.owner_ne_k3_manh)
#     a += +0.3 * min_max(board.owner_ne_k15_ones)
#     a += +1 * min_max(board.tile_live_k3_manh)
#     a += -1 * min_max(board.tile_dead_k3_manh)
#     a += +1 * min_max(board.units_op_k3_manh)

#     legal_moves = board.can_spawn
#     scores = a * np.where(legal_moves, 1, np.nan)
#     if np.all(np.isnan(scores)):
#         return ["SPAWN 1 0 0"]
#     max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
#     row, col = max_indices
#     return [f"SPAWN {k} {col} {row}"]

def get_spawn_2(board, k=1):
    min_max = lambda x: (x - x.min()) / (x.max() - x.min())
    new_spawns = np.full_like(board.scrap_amount, 0, dtype=np.float64)
    actions = []

    for i in range(k):
        legal_moves = board.can_spawn * (new_spawns == 0)
        if legal_moves.sum() == 0:
            break

        a = np.full_like(board.scrap_amount, 0, dtype=np.float64)

        a += +1 * min_max(board.scrap_amount_k3_manh)
        a += +9 * min_max(board.owner_op_k3_manh)
        a += -5 * min_max(board.owner_me_k3_manh)
        a += +5 * min_max(board.owner_ne_k3_manh)
        a += +1 * min_max(board.tile_live_k3_manh)
        a += -1 * min_max(board.tile_dead_k3_manh)
        a += +1 * min_max(board.units_op_k3_manh)
        a += -2 * min_max(board.units_me_k3_manh)
        if i > 0:
            a += -2 * min_max(convolve2d(new_spawns, board.manhattan_kernel, mode='same').astype(int))

        scores = a * np.where(legal_moves, 1, np.nan)
        max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
        row, col = max_indices
        new_spawns[row, col] = 1
        actions.append(f"SPAWN 1 {col} {row}")
    
    return actions



class InputOutput:

    def __init__(self, filename: str = "unittest.txt"):
        self.input = input_stream(filename) if dev else input
        self.input_log = []
        self.output_log = []

    def get_input(self, n: int = 1) -> None:
        self.input_log += [self.input() for _ in range(n)]
        lst = [i.split() for i in self.input_log[-n:]]
        return [int(item) for sublist in lst for item in sublist]

    def send_action(self, actions: list):
        message = ";".join(action.__str__() for action in actions)
        self.output_log.append(message)
        print(f"{message}")

# TODO
# capture and print the spawn max, min, mean & std (for threshold tuning)
# 5x5 kernel (at least for owner) for bigger maps
# spawn if op units > me units

def main():
    io = InputOutput('right.txt')
    col, row = io.get_input()
    weights = {
        'scrap_amount_k3_manh': +1,
        'owner_op_k3_manh':     +4,
        'owner_me_k3_manh':     -2,
        'owner_ne_k3_manh':     +2,
        'tile_live_k3_manh':    +1,
        'tile_dead_k3_manh':    -1,
        'units_op_k3_manh':     +1,
        'units_me_k3_manh':     -1,
    }

    while True:
        try:
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
            board = Board(input_array)
            actions = []
            actions += get_move_actions(board, weights)
            if my_matter >= 10 and board.units_op.sum() > 0:
                if board.can_build.sum() > 0:
                    if (max(1, board.recycler_me.sum()) / board.units_me.sum()) < 0.25:
                        actions += get_build_random(board)
                if board.can_spawn.sum() > 0:
                    if my_matter >= 10:
                        actions += get_spawn_2(board, k=(my_matter) // 10)
            io.send_action(actions)
            if verbose:
                debug('input_log')
                debug('\n'.join(io.input_log))
                debug('output_log')
                debug('\n'.join(io.output_log))
        except EOFError:
            debug("EOF")
            break

if __name__ == '__main__':
    main()

