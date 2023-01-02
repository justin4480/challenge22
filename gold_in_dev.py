dev = False if __file__ == '/tmp/Answer.py' else True

import sys
# from itertools import repeat
# import concurrent.futures
from collections import Counter
import numpy as np
from scipy.signal import fftconvolve
from scipy import ndimage
from time import perf_counter
# import matplotlib.pyplot as plt
# import seaborn as sns


verbose = False
print_input_logs = True

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
        self._input_array[:, :, 1] = np.where(self._input_array[:, :, 1] == 0, 2, self._input_array[:, :, 1])
        self.kernel_3_3_ones = np.ones((3, 3))
        self.kernel_manhattan = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]])
        self.kernel_global = get_filter(
            input_array.shape[0],
            input_array.shape[1],
            p=p)
        self._cached_properties = {}
        self.island_number = 0  # zero is all
        self.island_mask = np.ones((self._input_array.shape[0], self._input_array.shape[1]), dtype=np.int8)
        self.island = ndimage.label(
            # input=((self._input_array[:, :, 0] > 0) * (self._input_array[:, :, 3] == 0)),
            input=self._input_array[:, :, 0] > 0,
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
        self.active_island_stats = dict(sorted({
            island: {
                'tiles': np.sum((self.island == island) * self.tile_live),
                'units': {
                    'delta': np.sum((self.island == island) * self.units_me)
                                - np.sum((self.island == island) * self.units_op),
                    'me': np.sum((self.island == island) * self.units_me),
                    'op': np.sum((self.island == island) * self.units_op),
                },
                'owner': {
                    'delta': np.sum((self.island == island) * self.owner_me)
                                - np.sum((self.island == island) * self.owner_op),
                    'me': np.sum((self.island == island) * self.owner_me),
                    'op': np.sum((self.island == island) * self.owner_op),
                },
            }
            for island in self.island_active if island != 0
        }.items(), key=lambda item: item[1]['tiles'], reverse=True))


    def set_island_number(self, island_number):
        self.island_number = island_number
        self.island_mask = np.equal(self.island, self.island_number).astype(np.int8)

    def reset_island_number(self):
        self.island_number = 0
        self.island_mask = np.ones((self._input_array.shape[0], self._input_array.shape[1]), dtype=np.int8)

    def cached(self, propery_name, func):
        if propery_name not in self._cached_properties:
            self._cached_properties[propery_name] = {}
        if self.island_number not in self._cached_properties[propery_name]:
            if verbose:
                debug(f'updating cache for island {self.island_number} propery {propery_name}')
            self._cached_properties[propery_name][self.island_number] = func
        return self._cached_properties[propery_name][self.island_number]

    def add_build(self, row, col):
        self._input_array[row, col, 3] = 1
        self._input_array[row, col, 5] = 0

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
        func = self.can_spawn * np.isin(self.island, list(self.island_active)) * (self.recycler == 0)
        # func = self.island_shared * self.can_spawn * self.island_mask
        return self.cached('should_spawn', func)

    @property
    def tile_live(self):
        func = (self.scrap_amount > 0) * (self.recycler == 0) * self.island_mask
        return self.cached('tile_live', func)

    @property
    def tile_live_inc_recycler_fate(self):
        func = (self.scrap_amount > 0) * (self.recycler == 0) * (self.in_range_of_recycler == 0) * self.island_mask
        return self.cached('tile_live', func)

    @property
    def owner_me(self):
        func = (self.owner == 1) * self.island_mask
        return self.cached('owner_me', func)

    @property
    def owner_op(self):
        func = (self.owner == 2) * self.island_mask
        return self.cached('owner_op', func)

    @property
    def owner_ne(self):
        func = (self.owner == -1) * self.island_mask
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
    def tile_live_global(self):
        func = fftconvolve(self.tile_live * self.island_mask, self.kernel_global, mode='same')
        return self.cached('tile_live_global', func)

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
    def units_op_local(self):
        func = fftconvolve(self.units_op * self.island_mask, self.kernel_manhattan, mode='same')
        return self.cached('units_op_local', func)

    @property
    def units_me_local(self):
        func = fftconvolve(self.units_me * self.island_mask, self.kernel_manhattan, mode='same')
        return self.cached('units_me_local', func)

    @property
    def center_h_global(self):
        row, col = self._input_array[:, :, 0].shape
        array = np.zeros(self._input_array[:, :, 0].shape)
        array[row // 2, :] = 1
        return fftconvolve(array * self.island_mask, self.kernel_global, mode='same')

    @property
    def center_v_global(self):
        row, col = self._input_array[:, :, 0].shape
        array = np.zeros(self._input_array[:, :, 0].shape)
        array[:, col // 2] = 1
        return fftconvolve(array * self.island_mask, self.kernel_global, mode='same')

    # def print_islands(self):
    #     sns.heatmap(self.island, square=True, cbar=False, linecolor='white',
    #                 linewidths=.1, annot=True, fmt='d').set_title('Islands')

    # def print_property(self, property, include_island=True):
    #     if include_island:
    #         fig, ax = plt.subplots(2, sharex=True)
    #         sns.heatmap(self.island, square=True, cbar=False, linecolor='white',
    #                     linewidths=.1, annot=True, fmt='d', ax=ax[0]).set_title('Islands')
    #         sns.heatmap(getattr(self, property), square=True, cbar=False,
    #                     linecolor='white', linewidths=.1, annot=True, fmt='.1f', ax=ax[1]
    #                     ).set_title(f'island #{self.island_number} property: {property}')
    #     else:
    #         sns.heatmap(getattr(self, property), square=True, cbar=False,
    #                     linecolor='white', linewidths=.1, annot=True, fmt='.1f'
    #                     ).set_title(f'island #{self.island_number} property: {property}')

    # def print_custom(self, array, include_island=True):
    #     if include_island:
    #         fig, ax = plt.subplots(2, sharex=True)
    #         sns.heatmap(self.island, square=True, cbar=False, linecolor='white',
    #                     linewidths=.1, annot=True, fmt='d', ax=ax[0]).set_title('Islands')
    #         sns.heatmap(array, square=True, cbar=False,
    #                     linecolor='white', linewidths=.1, annot=True, fmt='.1f', ax=ax[1]
    #                     ).set_title(f'island #{self.island_number} property: {property}')
    #     else:
    #         sns.heatmap(array, square=True, cbar=False,
    #                     linecolor='white', linewidths=.1, annot=True, fmt='.1f'
    #                     ).set_title(f'island #{self.island_number} property: {property}')


def get_move_actions(board, weights):
    perf_start = perf_counter()
    actions = []

    islands = list(board.island_active)
    islands.remove(0)
    for island in islands:

        board.set_island_number(island)

        a = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
        a[board.tile_live == False] = np.nan
        a_padded = np.pad(a, 1, mode='constant', constant_values=np.nan)
        mask = np.array([[np.nan,1,np.nan], [1,np.nan,1], [np.nan,1,np.nan]])

        indices = np.where(board.units_me)
        for row, col in zip(indices[0], indices[1]):
            scores = a_padded[row:row+3, col:col+3] * mask
            if np.isnan(scores).all():
                continue
            row_offset, col_offset = [x - y for x, y in zip(np.unravel_index(np.nanargmax(scores), scores.shape), (1, 1))]
            actions.append(f"MOVE 1 {col} {row} {col + col_offset} {row + row_offset}")
    
    board.reset_island_number()
    debug(f'get_move_actions: {round((perf_counter() - perf_start) * 1000, 1)}')
    return actions





class Build:
    def __init__(self, board, weights, matter, max_build=1):
        self.board = board
        self.weights = weights
        self.matter = matter
        self.max_build = min(max_build, self.matter // 10)
        self._actions = []
        self.debug = True

    @property
    def island_numbers_can_build_on(self):
        return np.unique(self.board.can_build * self.board.island)

    @property
    def actions(self):
        if len(self.dominating_island) == 0:
            self._actions += self.get_best_build_position(self.best_island)
        else:
            for island in self.dominating_island:
                self._actions += self.get_best_build_position(island)
        for action in self._actions[:self.max_build]:
            _, col, row = action.split()
            self.board.add_build(int(row), int(col))
        return self._actions[:self.max_build]

    @property
    def dominating_island(self):
        return [k for k, v in self.board.active_island_stats.items()
            if v['tiles'] >= 1 and v['units']['delta'] > 0]

    @property
    def best_island(self):
        return list(dict(sorted({
            k: v['units']['delta'] for k, v in self.board.active_island_stats.items()
        }.items(), key=lambda item: item[1], reverse=True)).keys())[0]

    def get_best_build_position(self, island):
        self.board.set_island_number(island)
        c1 = fftconvolve(self.board.tile_live, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='same') * self.board.tile_live
        we = fftconvolve(c1, np.array([[1/3, 1/2, 1/3]]), mode='same') == 3
        ns = fftconvolve(c1, np.array([[1/3],[1/2],[1/3]]), mode='same') == 3
        d1 = fftconvolve(c1+1, np.array([[1,0,0],[0,1/4,0],[0,0,1]]), mode='same') == 3
        d2 = fftconvolve(c1+1, np.array([[0,0,1],[0,1/4,0],[1,0,0]]), mode='same') == 3
        build_targets = (
            fftconvolve(
                (((self.board.units_me + self.board.owner_me) * (self.board.units_op == 0)) > 0),
                self.board.kernel_global,
                mode='same',
            ) * self.board.can_build
        )
        build_targets_with_chokepoints = build_targets * (we | ns | d1 | d2)
        build_targets = build_targets_with_chokepoints if build_targets_with_chokepoints.sum() else build_targets
        my_density = min_max(self.board.owner_me_global + self.board.owner_ne_global)
        enemy_density = min_max(self.board.units_op_global + self.board.owner_op_global)
        boundary = my_density * enemy_density
        best_build = build_targets * boundary * self.board.tile_live
        row, col = np.unravel_index(np.nanargmax(best_build), best_build.shape)
        self.board.reset_island_number()
        return [f"BUILD {col} {row}"]



class Spawn:
    def __init__(self, board, weights, matter):
        self.board = board
        self.weights = weights
        self.matter = matter
        self.max_spawns = matter // 10
        self.priority_actions = []
        self.secondary_actions = []
        self._actions = []
        self.debug = True
        self.max_spawns_per_island = 5
        self.max_spawns_per_tile = 3

    @property
    def island_numbers_can_spawn_on(self):
        return np.unique(self.board.can_spawn * self.board.island)

    @property
    def actions(self):
        for island, amount in self.spawn_priority.items():
            if island in self.island_numbers_can_spawn_on:
                self.priority_actions += self.get_best_spawn_position(island, min(self.max_spawns_per_island, amount))
        # self.priority_actions += list(dict.fromkeys(self.priority_actions))  # remove duplicates
        self.secondary_actions = self.get_best_spawn_position(island=None, k=self.max_spawns)
        self._actions = self.priority_actions + self.secondary_actions
        if self.debug:
            debug(f'{self.spawn_where_my_units_count_is_zero=}')
            debug(f'{self.spawn_where_delta_not_positive=}')
            debug(f'{self.priority_actions=}')
            debug(f'{self.secondary_actions=}')
            debug(f'return {self._actions[:self.max_spawns]}')
        top_k = self._actions[:self.max_spawns]
        collapsed_results = [element.replace('SPAWN 1', f'SPAWN {min(count, self.max_spawns_per_tile)}') for element, count in Counter(top_k).items()]
        return collapsed_results
        
    @property
    def spawn_where_my_units_count_is_zero(self):
        return {
            k: 1
            for k, v in self.board.active_island_stats.items()
            if v['tiles'] >= 1 and v['units']['me'] == 0
        }
        
    @property
    def spawn_where_delta_not_positive(self):
        return {
            k: (v['units']['delta'] * -1) + 1
            for k, v in self.board.active_island_stats.items()
            if v['tiles'] >= 1 and v['units']['delta'] <= 0
        }
        
    @property
    def spawn_priority(self):
        return {k: v for d in [
            self.spawn_where_my_units_count_is_zero,
            self.spawn_where_delta_not_positive,
        ] for k, v in d.items()}

    @property
    def spawn_secondary_island_list(self):
        return dict(sorted({
            k: v['units']['delta'] / v['tiles']
            for k, v in self.board.active_island_stats.items()
            if v['tiles'] >= 1
        }.items(), key=lambda item: item[1], reverse=False))

    def get_best_spawn_position(self, island, k=1):
        new_spawns = np.zeros(self.board.should_spawn.shape, dtype=np.float16)
        if island is None:
            self.board.reset_island_number()
        else:
            self.board.set_island_number(island)
        func_actions = []
        scores = np.nansum([weight * min_max(getattr(self.board, feature))
                            for feature, weight in self.weights.items()], axis=0)
        scores[self.board.should_spawn == False] = np.nan
        for i in range(k):
            new_spawn_penalty = -3 * min_max(fftconvolve(new_spawns, self.board.kernel_global, mode='same'))
            row, col = np.unravel_index(np.nanargmax(scores + new_spawn_penalty), scores.shape)
            new_spawns[row, col] += 1
            func_actions.append(f"SPAWN 1 {col} {row}")
        self.board.reset_island_number()
        return func_actions


def get_build(board, weights, k=1):
    perf_start = perf_counter()
    actions = []
    scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
    scores[board.should_build == False] = np.nan
    for i in range(k):
        row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
        actions.append(f"BUILD {col} {row}")
        # scores[row, col] = np.nan
    debug(f'get_build: {round((perf_counter() - perf_start) * 1000, 1)}')
    return actions


def get_build_new(board):
    perf_start = perf_counter()
    board.reset_island_number()
    c1 = fftconvolve(board.tile_live, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='same') * board.tile_live
    we = fftconvolve(c1, np.array([[1/3, 1/2, 1/3]]), mode='same') == 3
    ns = fftconvolve(c1, np.array([[1/3],[1/2],[1/3]]), mode='same') == 3
    d1 = fftconvolve(c1+1, np.array([[1,0,0],[0,1/4,0],[0,0,1]]), mode='same') == 3
    d2 = fftconvolve(c1+1, np.array([[0,0,1],[0,1/4,0],[1,0,0]]), mode='same') == 3
    build_targets = (
        fftconvolve(
            (((board.units_me + board.owner_me) * (board.units_op == 0)) > 0),
            board.kernel_global,
            mode='same',
        ) * (we | ns | d1 | d2) * board.can_build
    )
    islands = set(np.unique(np.where(build_targets>0, board.island, np.nan))).intersection(board.island_active)
    results = []
    for i in islands:
        board.set_island_number(i)
        my_density = min_max(board.owner_me_global + board.owner_ne_global)
        enemy_density = min_max(board.units_op_global + board.owner_op_global)
        boundary = my_density * enemy_density
        best_build = build_targets * boundary * board.tile_live
        results.append((np.nanmax(best_build), np.unravel_index(np.nanargmax(best_build), best_build.shape)))
    board.reset_island_number()
    debug(f'get_build_new: {round((perf_counter() - perf_start) * 1000, 1)}')
    if islands:
        results.sort()
        row, col = results[-1:][0][1]
        board.reset_island_number()
        return [f"BUILD {col} {row}"]
    else:
        my_density = min_max(board.owner_me_global + board.owner_ne_global)
        enemy_density = min_max(board.units_op_global + board.owner_op_global)
        boundary = my_density * enemy_density
        best_build = boundary * board.can_build
        row, col = np.unravel_index(np.nanargmax(best_build), best_build.shape)
        board.reset_island_number()
        return [f"BUILD {col} {row}"]


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

# SPAWN - add premimum if tile_will_die_in_t_2 (to get unit into enemy zone before blocked)
# BUILD - net_tile_ownership_if_built


def main():
    io = InputOutput('issue3.txt')
    col, row = io.get_input()
    perf = []
    # move_weights = {
    #     'owner_op_global':    +4,
    #     'owner_me_global':    -2,
    #     'owner_ne_global':    +2,
    #     'units_op_global':    +3,
    #     'units_me_global':    -3,
    # }
    # spawn_weights = {
    #     'owner_op_local':     +2,
    #     'owner_ne_local':     +1,
    # }
    # build_weights = {
    #     'tile_live_global':  -4,
    #     'owner_op_global':   +4,
    #     'owner_ne_global':   +4,
    #     'owner_op_local':    -4,
    #     'owner_ne_local':    -1,
    #     'owner_me_local':    +1,
    #     'units_op_local':    -5,
    #     'units_me_local':    -2,
    #     'center_v_global':   +9,
    #     'center_h_global':   +3,
    # }
    move_weights = {
        'owner_op_global':    +20,
        'owner_me_global':    -10,
        'owner_ne_global':    +10,
        'units_op_global':    +15,
        'units_me_global':    -15,
    }
    spawn_weights = {
        'owner_op_global':    +40,
        'owner_ne_global':    +10,
        'units_me_global':    -10,
    }
    build_weights = {
        'scrap_amount_local': +20,
        'owner_op_global':    +20,
        'owner_ne_global':    +10,
        'units_me_global':    -20,
    }


    while True:
        try:
            perf_start = perf_counter()
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col), dtype=np.int16).reshape(row, col, 7)
            board = Board(input_array)
            actions = []
            stopclock = perf_counter()
            actions += get_move_actions(board, move_weights)
            debug(f'  - MOVE: {round((perf_counter() - stopclock) * 1000, 1)}ms')
            stopclock = perf_counter()
            if board.units_op.sum() > 0:
                if my_matter >= 10 and board.should_build.sum() > 0:
                    if board.recycler_me.sum() / board.should_build.sum() < 0.15:
                        stopclock = perf_counter()
                        build_actions = Build(board, build_weights, my_matter, max_build=1).actions
                        debug(f'  - BUILD: {round((perf_counter() - stopclock) * 1000, 1)}ms')
                        stopclock = perf_counter()
                        actions += build_actions
                        my_matter -= 10
                if my_matter >= 10 and board.should_spawn.sum() > 0:
                    stopclock = perf_counter()
                    spawn_actions = Spawn(board, spawn_weights, my_matter).actions
                    debug(f'  - SPAWN: {round((perf_counter() - stopclock) * 1000, 1)}ms')
                    stopclock = perf_counter()
                    actions += spawn_actions
                    my_matter -= len(actions) * 10
            perf_end = perf_counter()
            perf.append(round((perf_end - perf_start) * 1000, 1))
            actions += [f'MESSAGE {perf[-1:][0]}ms ({round(np.mean(perf), 1)}ms)']
            io.send_action(actions)
        except EOFError:
            debug("EOF")
            break

if __name__ == '__main__':
    main()

