dev = False if __file__ == '/tmp/Answer.py' else True

import sys
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
# import matplotlib.pyplot as plt
# import seaborn as sns

verbose = False
print_input_logs = False

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


min_max = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)


def get_filter(rows, cols, p):
    n = max(rows, cols) * 2 + 1
    a = np.concatenate([np.arange(n//2, 0, -1), [0.25], np.arange(1, n//2+1, 1)], axis=0)
    X, Y = np.meshgrid(a, a)
    filter = p ** (X+Y)
    filter[n//2 - 1: n//2 + 2, n//2] = 3
    filter[n//2, n//2 - 1: n//2 + 2] = 3
    return filter


class Board:

    def __init__(self, input_array, p=0.85):
        self._input_array = input_array
        self.kernel_3_3_ones = np.ones(
            (3, 3))
        self.kernel_manhattan = np.array(
            [[0,1,0],
             [1,1,1],
             [0,1,0]])
        self.kernel_local = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]])
        self.kernel_global = get_filter(
            input_array.shape[0],
            input_array.shape[1],
            p=p)
        
    @property
    def tile_cluster_shared(self):
        tile_cluster = ndimage.label(
            input=self.tile_live.astype(int),
            structure=self.kernel_manhattan,
        )[0]
        shared_clusters = (
            set(np.unique(self.owner_op * tile_cluster))
            .intersection(set(np.unique(self.owner_me * tile_cluster)))
        )
        return np.isin(tile_cluster, list(shared_clusters))

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
    def should_build(self):
        return self.tile_cluster_shared * self.can_build

    @property
    def can_spawn(self):
        return self._input_array[:, :, 5]

    @property
    def should_spawn(self):
        return self.tile_cluster_shared * self.can_spawn

    @property
    def in_range_of_recycler(self):
        return self._input_array[:, :, 6]

    @property
    def is_grass(self):
        return self.scrap_amount == 0

    @property
    def scrap_amount_local(self):
        return convolve2d(self.scrap_amount, self.kernel_local, mode='same')

    @property
    def owner_me(self):
        return self.owner == 1

    @property
    def owner_me_local(self):
        return convolve2d(self.owner_me, self.kernel_local, mode='same')

    @property
    def owner_me_global(self):
        return convolve2d(self.owner_me, self.kernel_global, mode='same')

    @property
    def owner_op(self):
        return self.owner == 0

    @property
    def owner_op_3_3_sq(self):
        return convolve2d(self.owner_op, self.kernel_3_3_ones, mode='same')

    @property
    def owner_op_local(self):
        return convolve2d(self.owner_op, self.kernel_local, mode='same')

    @property
    def owner_op_global(self):
        return convolve2d(self.owner_op, self.kernel_global, mode='same')

    @property
    def owner_ne(self):
        return self.owner == -1

    @property
    def owner_ne_local(self):
        return convolve2d(self.owner_ne, self.kernel_local, mode='same')

    @property
    def owner_ne_global(self):
        return convolve2d(self.owner_ne, self.kernel_global, mode='same')

    @property
    def tile_live(self):
        return (self.scrap_amount > 0) * (self.recycler == 0)

    @property
    def tile_live_global(self):
        return convolve2d(self.tile_live, self.kernel_global, mode='same')

    @property
    def tile_dead(self):
        return self.tile_live == 0

    @property
    def tile_dead_global(self):
        return convolve2d(self.tile_dead, self.kernel_global, mode='same')

    @property
    def units_op(self):
        return self.units * (self.owner == 0)

    @property
    def units_op_local(self):
        return convolve2d(self.units_op, self.kernel_local, mode='same')

    @property
    def units_op_global(self):
        return convolve2d(self.units_op, self.kernel_global, mode='same')

    @property
    def units_me(self):
        return self.units * (self.owner == 1)

    @property
    def units_me_local(self):
        return convolve2d(self.units_me, self.kernel_local, mode='same')

    @property
    def units_me_global(self):
        return convolve2d(self.units_me, self.kernel_global, mode='same')

    @property
    def recycler_me(self):
        return self._input_array[:, :, 3] * (self.owner == 1)


def get_move_actions(board, weights):
    results = []
    actions = []

    a = np.full_like(board.scrap_amount, 0, dtype=np.float64)

    for feature, weight in weights.items():
        a += weight * min_max(getattr(board, feature))

    for row, col in np.argwhere(board.units_me):
        current_unit = np.zeros_like(a)
        current_unit[row, col] = 1
        units_me_k3_manh_excl_current_unit = convolve2d(
            board.units * (board.owner == 1) * (current_unit == 0),
            board.kernel_local, mode='same'
        ).astype(int)

        a += -2 * min_max(units_me_k3_manh_excl_current_unit)

        legal_moves = board.tile_live * convolve2d(current_unit, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='same')
        scores = a * np.where(legal_moves, 1, np.nan)
        if np.all(np.isnan(scores)):
            continue
        max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
        actions.append(f"MOVE 1 {col} {row} {max_indices[1]} {max_indices[0]}")
        results.append({'row_col': (row, col), 'max_indices': max_indices, 'max_value': max_value})

    return actions


def get_spawn(board, weights, k=1):
    new_spawns = np.full_like(board.scrap_amount, 0, dtype=np.float64)
    actions = []

    for i in range(k):
        # (board.units_me == 0) is to prevent spawning on top of own units
        legal_moves = (
            1
            * (board.can_spawn == 1)
            * (new_spawns == 0)
            * (board.recycler == 0)
            * (board.units_me == 0)
            * (board.tile_cluster_shared == 1)
        )
        if legal_moves.sum() == 0:
            break

        a = np.full_like(board.scrap_amount, 0, dtype=np.float64)

        for feature, weight in weights.items():
            a += weight * min_max(getattr(board, feature))

        if i > 0:
            a += -2 * min_max(convolve2d(new_spawns, board.kernel_local, mode='same').astype(int))

        scores = a * np.where(legal_moves, 1, np.nan)
        max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
        row, col = max_indices
        new_spawns[row, col] = 1
        actions.append(f"SPAWN 1 {col} {row}")
    
    return actions


def get_build(board):
    legal_moves = board.can_build * board.tile_cluster_shared
    a = (board.owner_op_3_3_sq) + ((board.scrap_amount_local * board.in_range_of_recycler == 0) / 100)
    scores = a * np.where(legal_moves, 1, np.nan)
    max_value, max_indices = np.nanmax(scores), np.unravel_index(np.nanargmax(scores), scores.shape)
    row, col = max_indices
    return [f"BUILD {col} {row}"]


class InputOutput:

    def __init__(self, filename: str = "unittest.txt"):
        self.input = input_stream(filename) if dev else input
        self.input_log = []
        self.output_log = []

    def get_input(self, n: int = 1) -> None:
        if False:
            lst = [self.input().split() for _ in range(n)]
            return [int(item) for sublist in lst for item in sublist]
        else:
            self.input_log += [self.input() for _ in range(n)]
            if print_input_logs:
                [debug(log) for log in self.input_log[-n:]]
            lst = [i.split() for i in self.input_log[-n:]]
            return [int(item) for sublist in lst for item in sublist]

    def send_action(self, actions: list):
        if len(actions) == 0:
            actions = ["WAIT"]
        message = ";".join(action.__str__() for action in actions)
        self.output_log.append(message)
        print(f"{message}")

# TODO
# - capture and print the spawn max, min, mean & std (for threshold tuning)
# - spawn if op units > me units
# - Agression - Build / spawn rate dynamic to my tiles / op tiles 
# - Spawn - Edge of op else edge of ne

# - islands - Consider unique boards / games per island (Use masks? * layer / cluster 3 )
# - island domination:
#       - dominated - spawn more, build less
#       - dominating - Spawn less, build more

# - On island , Move to switch to path based only
# - Remove islands entirely from conv gradients or make unique gradients for each cluster 


def main():
    io = InputOutput('right.txt')
    col, row = io.get_input()
    # weights = {
    #     'scrap_amount_local': +1,
    #     'owner_op_local':     +40,
    #     'owner_op_global':    +4,
    #     'owner_me_local':     -20,
    #     'owner_me_global':    -2,
    #     'owner_ne_local':     +20,
    #     'owner_ne_global':    +2,
    #     'tile_live_global':   +1,
    #     'tile_dead_global':   -1,
    #     'units_op_local':     +10,
    #     'units_op_global':    +1,
    #     'units_me_local':     -10,
    #     'units_me_global':    -1,
    # }
    # weights = {
    #     'scrap_amount_local': +1,
    #     'owner_op_global':    +20,
    #     'owner_me_global':    -10,
    #     'owner_ne_global':    +10,
    #     'tile_live_global':   +1,
    #     'tile_dead_global':   -1,
    #     'units_op_global':    +1,
    #     'units_me_global':    -1,
    # }
    weights = {
        'scrap_amount_local': +1,  # 380 before ~ 420 after
        'owner_op_global':    +20,
        'owner_me_global':    -10,
        'owner_ne_global':    +10,
        'tile_live_global':   +1,  # 450 after
        'tile_dead_global':   -1,  # 450 after
        'units_op_global':    +15,
        'units_me_global':    -15,
    }


    while True:
        try:
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
            board = Board(input_array)
            actions = []
            actions += get_move_actions(board, weights)
            if board.units_op.sum() > 0:
                if my_matter >= 10 and board.should_build.sum() > 0:
                    # if (board.units_me.sum() / max(1, board.recycler_me.sum())) > 6:
                    if board.recycler_me.sum() / (board.owner_me * board.tile_cluster_shared).sum() < 0.1:
                        actions += get_build(board)
                        my_matter -= 10
                if my_matter >= 10 and board.should_spawn.sum() > 0:
                    k = (my_matter) // 10
                    actions += get_spawn(board, weights, k)
                    my_matter -= k * 10
            io.send_action(actions)
        except EOFError:
            debug("EOF")
            break

if __name__ == '__main__':
    main()

