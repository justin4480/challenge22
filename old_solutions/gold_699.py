dev = False if __file__ == '/tmp/Answer.py' else True

import sys
from collections import Counter
import numpy as np
from scipy.signal import fftconvolve
from scipy import ndimage
from time import perf_counter
import pandas as pd
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
# import seaborn as sns

verbose = True
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
    filter[n//2 - 1: n//2 + 2, n//2] = 2
    filter[n//2, n//2 - 1: n//2 + 2] = 2
    return filter


class Board:

    def __init__(self, input_array, p=0.95):
        self._input_array = input_array
        self.height, self.width = input_array.shape[0], input_array.shape[1]
        self._input_array[:, :, 1] = np.where(self._input_array[:, :, 1] == 0, 2, self._input_array[:, :, 1])
        self.kernel_3_3_ones = np.ones((3, 3))
        self.kernel_manhattan = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]])
        self.kernel_global = get_filter(
            input_array.shape[0],
            input_array.shape[1],
            p=(74 + (2 * (self.width-12))) / 100)
        self._cached_properties = {}
        self.island_number = 0  # zero is all
        self.island_mask = np.ones((self._input_array.shape[0], self._input_array.shape[1]))
        self.island = ndimage.label(
            input=((self._input_array[:, :, 0] > 0) * (self._input_array[:, :, 3] == 0)),
            structure=self.kernel_manhattan,
        )[0]
        self.island_presence_me = set(np.unique(self.owner_me * self.island))
        self.island_presence_op = set(np.unique(self.owner_op * self.island))
        self.island_presence_ne = set(np.unique(self.owner_ne * self.island))
        self.island_shared = set(self.island_presence_me).intersection(set(self.island_presence_op))
        self.island_captured_me = self.island_presence_me - self.island_shared
        self.island_captured_op = self.island_presence_op - self.island_shared
        self.island_complete = self.island_captured_op | self.island_captured_me.difference(self.island_presence_ne)
        self.island_active = set(np.unique(self.island)).difference(self.island_complete)
        self.zones = self.get_zones()
        self.me_right = self.zones[:, :self.zones.shape[1] // 2].sum() > self.zones[:, self.zones.shape[1] // 2:].sum()
        self.kernel_defensive = (np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])if self.me_right else
                                 np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]))

    def get_zones(self):
        df = pd.DataFrame(
            data=[(row, col, self.owner_me[row, col])
                for row in range(self.owner_me.shape[0])
                for col in range(self.owner_me.shape[1])],
            columns=['row', 'col', 'owner_me'])
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(X=df[['row', 'col']], y=df.owner_me) 
        return lr.predict(df[['row', 'col']]).reshape(self.owner_me.shape)

    def set_island_number(self, island_number):
        self.island_number = island_number
        self.island_mask = (self.island == self.island_number)

    def reset_island_number(self):
        self.island_number = 0
        self.island_mask = np.ones((self._input_array.shape[0], self._input_array.shape[1]))

    def cached(self, propery_name, func):
        if propery_name not in self._cached_properties:
            self._cached_properties[propery_name] = {}
        if self.island_number not in self._cached_properties[propery_name]:
            if verbose:
                debug(f'updating cache for island {self.island_number} propery {propery_name}')
            self._cached_properties[propery_name][self.island_number] = func
        return self._cached_properties[propery_name][self.island_number]

    @property
    def live_shared_tile(self):
        func = (self.tile_live * np.isin(self.island, list(self.island_shared)))
        return self.cached('live_shared_tile', func)

    @property
    def live_active_tile(self):
        func = (self.tile_live * np.isin(self.island, list(self.island_active)))
        return self.cached('live_active_tile', func)

    @property
    def scrap_amount(self):
        func = self._input_array[:, :, 0] * self.island_mask
        return self.cached('scrap_amount', func)

    @property
    def owner(self):
        func = self._input_array[:, :, 1] * self.island_mask
        return self.cached('owner', func)

    @property
    def units(self):
        func = self._input_array[:, :, 2] * self.island_mask
        return self.cached('units', func)

    @property
    def recycler(self):
        func = self._input_array[:, :, 3] * self.island_mask
        return self.cached('recycler', func)

    @property
    def can_build(self):
        func = self._input_array[:, :, 4] * self.island_mask
        return self.cached('can_build', func)

    @property
    def can_spawn(self):
        func = self._input_array[:, :, 5] * self.island_mask
        return self.cached('can_spawn', func)

    @property
    def in_range_of_recycler(self):
        func = self._input_array[:, :, 6] * self.island_mask
        return self.cached('in_range_of_recycler', func)

    # @property
    # def is_grass(self):
    #     func = (self.scrap_amount == 0) * self.island_mask
    #     return self.cached('is_grass', func)

    @property
    def should_build(self):
        func = self.can_build * np.isin(self.island, list(self.island_shared))
        # func = self.island_shared * self.can_build * self.island_mask
        return self.cached('should_build', func)

    @property
    def should_spawn(self):
        func = self.can_spawn * np.isin(self.island, list(self.island_active))
        # func = self.island_shared * self.can_spawn * self.island_mask
        return self.cached('should_spawn', func)

    @property
    def tile_live(self):
        func = (self.scrap_amount > 0) * (self.recycler == 0) * self.island_mask
        return self.cached('tile_live', func)

    @property
    def op_units_in_my_zone(self):
        func = self.units_op * self.zones
        return self.cached('op_units_in_my_zone', func)

    @property
    def owner_me(self):
        func = (self.owner == 1) * (self.in_range_of_recycler == 0) * self.island_mask
        return self.cached('owner_me', func)

    @property
    def owner_op(self):
        func = (self.owner == 2) * (self.in_range_of_recycler == 0) * self.island_mask
        return self.cached('owner_op', func)

    @property
    def owner_ne(self):
        func = (self.owner == -1) * (self.in_range_of_recycler == 0) * (self.tile_live == 1) * self.island_mask
        return self.cached('owner_ne', func)

    @property
    def units_op(self):
        func = self.units * (self.owner == 2) * self.island_mask
        return self.cached('units_op', func)

    @property
    def units_me(self):
        func = self.units * (self.owner == 1) * self.island_mask
        return self.cached('units_me', func)

    @property
    def recycler_me(self):
        func = self._input_array[:, :, 3] * (self.owner == 1) * self.island_mask
        return self.cached('recycler_me', func)

    @property
    def scrap_amount_local(self):
        func = fftconvolve(self.scrap_amount * self.island_mask, self.kernel_manhattan, mode='same')
        return self.cached('scrap_amount_local', func)

    # @property
    # def tile_live_global(self):
    #     func = fftconvolve(self.tile_live * self.island_mask, self.kernel_global, mode='same')
    #     return self.cached('tile_live_global', func)

    # @property
    # def zone_breached_manh(self):
    #     func = fftconvolve(self.zones * self.units_op, self.kernel_manhattan, mode='same').astype(int)
    #     return self.cached('zone_breached_manh', func)

    @property
    def zone_breach_op_unit_positions(self):
        func = (self.zones * self.units_op).astype(int)
        return self.cached('zone_breach_op_unit_positions', func)

    @property
    def zone_breached_deploy_defense(self):
        func = np.round(fftconvolve(self.zone_breach_op_unit_positions,
                                    self.kernel_defensive, mode='same'), 0).astype(int)
        return self.cached('zone_breached_deploy_defense', func)

    @property
    def owner_me_local(self):
        func = fftconvolve(self.owner_me * self.island_mask, self.kernel_manhattan, mode='same')
        return self.cached('owner_me_local', func)

    @property
    def owner_op_local(self):
        func = fftconvolve(self.owner_op * self.island_mask, self.kernel_manhattan, mode='same')
        return self.cached('owner_op_local', func)

    @property
    def owner_ne_local(self):
        func = fftconvolve(self.owner_ne * self.island_mask, self.kernel_manhattan, mode='same')
        return self.cached('owner_ne_local', func)

    @property
    def owner_me_global(self):
        func = fftconvolve(self.owner_me * self.island_mask, self.kernel_global, mode='same')
        return self.cached('owner_me_global', func)

    @property
    def owner_op_global(self):
        func = fftconvolve(self.owner_op * self.island_mask, self.kernel_global, mode='same')
        return self.cached('owner_op_global', func)

    @property
    def owner_ne_global(self):
        func = fftconvolve(self.owner_ne * self.island_mask, self.kernel_global, mode='same')
        return self.cached('owner_ne_global', func)

    # @property
    # def owner_op_3_3_sq(self):
    #     func = fftconvolve(self.owner_op * self.island_mask, self.kernel_3_3_ones, mode='same')
    #     return self.cached('owner_op_3_3_sq', func)

    @property
    def units_op_global(self):
        func = fftconvolve(self.units_op * self.island_mask, self.kernel_global, mode='same')
        return self.cached('units_op_global', func)

    @property
    def units_me_global(self):
        func = fftconvolve(self.units_me * self.island_mask, self.kernel_global, mode='same')
        return self.cached('units_me_global', func)

    @property
    def op_end_global(self):
        col = self.width-1 if self.me_right else 0
        op_end = np.zeros((self.height, self.width), dtype=int)
        op_end[:, col] = 1
        func = fftconvolve(op_end, self.kernel_global, mode='same')
        return self.cached('op_end_global', func)

def get_move_actions(board, weights):
    actions = []
    islands = list(board.island_active)
    islands.remove(0)
    for island in islands:
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
        scores[board.live_active_tile == False] = np.nan
        scores_padded = np.pad(scores, 1, mode='constant', constant_values=np.nan)
        mask = np.array([[np.nan,1,np.nan], [1,np.nan,1], [np.nan,1,np.nan]])
        indices = np.where(board.units_me * board.live_active_tile)
        for row, col in zip(indices[0], indices[1]):
            scores = scores_padded[row:row+3, col:col+3] * mask
            if np.isnan(scores).all():
                continue
            row_offset, col_offset = [x - y for x, y in zip(np.unravel_index(np.nanargmax(scores), scores.shape), (1, 1))]
            actions.append(f"MOVE 1 {col} {row} {col + col_offset} {row + row_offset}")
    board.reset_island_number()
    return actions

def get_spawn_actions(board, weights, k=1):
    actions = []
    islands = list(board.island_active)
    islands.remove(0)
    for island in islands:
        board.set_island_number(island)
        new_spawns = np.zeros(board.should_spawn.shape, dtype=np.float16)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
        scores[board.can_spawn == False] = np.nan
        scores[board.live_active_tile == False] = np.nan
        for i in range(k):
            if np.isnan(scores).all():
                break
            new_spawn_penalty = -5 * min_max(fftconvolve(new_spawns, board.kernel_global, mode='same'))
            row, col = np.unravel_index(np.nanargmax(scores + new_spawn_penalty), scores.shape)
            actions.append(f"SPAWN 1 {col} {row}")
            new_spawns[row, col] += 1
            scores[row, col] = np.nan
    board.reset_island_number()
    return actions

def get_build_actions(board, weights, k=1):
    actions = []
    islands = list(board.island_active)
    islands.remove(0)
    for island in islands:
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
        scores[board.can_build == False] = np.nan
        scores[board.live_shared_tile == False] = np.nan
        for i in range(k):
            if np.isnan(scores).all():
                break
            row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
            actions.append(f"BUILD {col} {row}")
            scores[row, col] = np.nan
    board.reset_island_number()
    return actions


class InputOutput:

    def __init__(self, filename: str = "unittest.txt"):
        self.input = input_stream(filename) if dev else input
        self.input_log = []
        self.output_log = []

    def get_input(self, n: int = 1) -> None:
        if print_input_logs:
            self.input_log += [self.input() for _ in range(n)]
            if print_input_logs:
                [debug(log) for log in self.input_log[-n:]]
            lst = [i.split() for i in self.input_log[-n:]]
            return [int(item) for sublist in lst for item in sublist]
        else:
            lst = [self.input().split() for _ in range(n)]
            return [int(item) for sublist in lst for item in sublist]

    def send_action(self, actions: list):
        if len(actions) == 0:
            actions = ["WAIT"]
        message = ";".join(action.__str__() for action in actions)
        if print_input_logs:
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
    io = InputOutput('p272_crash.txt')
    col, row = io.get_input()
    perf = []
    move_weights = {
        'op_end_global':    +5,
        'owner_me_local':   -1,
        'owner_ne_local':   +1,
        # 'owner_op_local':   +1,
        'owner_op_global':  +3,
        # 'owner_ne_global':  +2,
    }
    spawn_weights = {
        'op_end_global':    +5,
        'owner_op_local':   +4,
        'owner_ne_local':   +3,
        # 'units_me_global':  -1,
    }
    build_weights = {
        # 'owner_me_global':  -5,
        'units_op_global':  +1,
    }


    while True:
        try:
            perf_start = perf_counter()
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
            board = Board(input_array)
            actions = []
            if verbose:
                debug(f'\nIslands\n{"-"*40}')
                debug(board.island)
                debug(f'\nZones\n{"-"*40}')
                debug(board.zones.astype(int))
                debug(f'\nActive\n{"-"*40}')
                debug(board.live_active_tile.astype(int))
                debug(f'\nShared\n{"-"*40}')
                debug(board.live_shared_tile.astype(int))
                debug(f'\nBreaches\n{"-"*40}')
                debug(board.zone_breach_op_unit_positions)

            # deal with zone breaches
            verbose and debug(f'\nZone Breach actions ({my_matter})\n{"-"*40}')
            if board.zone_breach_op_unit_positions.sum() == 0:
                verbose and debug('No zone breaches')
            else:
                zone_breach_build = board.zone_breached_deploy_defense * (board.can_build == 1)
                zone_breach_spawn = board.zone_breached_deploy_defense * (board.can_build == 0) * board.can_spawn
                build_actions = [f'BUILD {col} {row}' for row, col in zip(*zone_breach_build.nonzero())]
                spawn_actions = [f'SPAWN 1 {col} {row}' for row, col in zip(*zone_breach_spawn.nonzero())]
                verbose and debug(f'{board.zone_breach_op_unit_positions.sum()} units breached my zone!')
                verbose and debug(f'{build_actions=}')
                verbose and debug(f'{spawn_actions=}')
                zone_breach_actions = build_actions + spawn_actions
                affordable_zone_breach_actions = zone_breach_actions[:my_matter // 10]
                verbose and debug(f'{affordable_zone_breach_actions=}')
                actions += affordable_zone_breach_actions
                my_matter -= len(affordable_zone_breach_actions) * 10

            # get MOVE actions
            verbose and debug(f'\nMOVE actions ({my_matter})\n{"-"*40}')
            if board.units_me.sum() > 0:
                move_actions = get_move_actions(board, move_weights)
                verbose and debug(f'{move_actions=}')
                actions += move_actions

            # get BUILD actions
            verbose and debug(f'\nBUILD actions ({my_matter})\n{"-"*40}')
            if my_matter < 10:
                verbose and debug(f'Not enought matter for BUILD actions')
            elif board.recycler_me.sum() / (1e-7 + (board.owner_me * board.live_active_tile).sum()) > .25:
                verbose and debug(f'More than 15% of my tiles are recyclers')
            else:
                build_actions = get_build_actions(board, build_weights)
                verbose and debug(f'{build_actions=}')
                actions += build_actions
                my_matter -= len(build_actions) * 10

            # get SPAWN actions
            verbose and debug(f'\nSPAWN actions ({my_matter})\n{"-"*40}')
            if my_matter < 10:
                verbose and debug(f'Not enought matter for BUILD actions')
            elif board.live_active_tile.sum() == 0:
                verbose and debug(f'No live active tiles to SPAWN to')
            else:
                spawn_actions = get_spawn_actions(board, spawn_weights, k=my_matter // 10)
                verbose and debug(f'{spawn_actions=}')
                actions += spawn_actions
                my_matter -= len(spawn_actions) * 10


            perf_end = perf_counter()
            perf.append(round((perf_end - perf_start) * 1000, 1))
            verbose and debug(f'\nFinal Actions ({my_matter})\n{"-"*40}')
            actions += [f'MESSAGE {perf[-1:][0]}ms ({round(np.mean(perf), 1)}ms)']
            io.send_action(actions)
        except EOFError:
            verbose and debug("EOF")
            break

if __name__ == '__main__':
    main()
