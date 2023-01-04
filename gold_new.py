dev = False if __file__ == '/tmp/Answer.py' else True

import sys
import numpy as np
import json
from scipy.signal import fftconvolve
from scipy import ndimage
from time import perf_counter
from sklearn.linear_model import LogisticRegression

verbose = -1
print_input_logs = True

def debug(message) -> None:
    if verbose > -1:
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

def get_kernel_global(rows, cols, p):
    n = max(rows, cols) * 2 + 1
    a = np.concatenate([np.arange(n//2, 0, -1), [0.25], np.arange(1, n//2+1, 1)], axis=0)
    X, Y = np.meshgrid(a, a)
    filter = p ** (X+Y)
    filter[n//2 - 1: n//2 + 2, n//2] = 2
    filter[n//2, n//2 - 1: n//2 + 2] = 2
    return filter


def get_kernel_horizonal_gradient(rows, cols, p):
    n = max(rows, cols) * 2
    a = np.concatenate([np.arange(n)], axis=0)
    X, Y = np.meshgrid(a, a)
    filter = p ** (X)
    return filter[:rows, :cols]


class Board:

    def __init__(self, input_array, kernel_global):
        self._input_array = input_array
        self.height, self.width = input_array.shape[0], input_array.shape[1]
        self._input_array[:, :, 1] = np.where(self._input_array[:, :, 1] == 0, 2, self._input_array[:, :, 1])

        self.kernel_3_3_ones = np.ones((3, 3))
        self.kernel_manhattan = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]])
        self.kernel_manhattan_vertical = np.array(
            [[0, 1, 0],
             [1, 1, 0],
             [0, 1, 0]])
        self.kernel_global = kernel_global
        self._cached_properties = {}
        self.island_number = -1
        self.island_mask = np.ones((self.height, self.width), dtype=bool)
        self.island = ndimage.label(
            input=((self._input_array[:, :, 0] > 0) * (self._input_array[:, :, 3] == 0)),
            structure=self.kernel_manhattan,
        )[0]
        self.island_dead = {0}
        self.island_presence_me = set(np.unique(self.owner_me * self.island)).difference(self.island_dead)
        self.island_presence_op = set(np.unique(self.owner_op * self.island)).difference(self.island_dead)
        self.island_presence_ne = set(np.unique(self.owner_ne * self.island)).difference(self.island_dead)
        
        # self.island_captured_me = self.island_presence_me.difference(self.island_presence_op)
        # self.island_captured_op = self.island_presence_op.difference(self.island_presence_me)
        # self.island_captured_ne = self.island_presence_ne.difference(self.island_presence_me).difference(self.island_presence_op)

        self.island_shared = set(self.island_presence_me).intersection(set(self.island_presence_op))
        self.island_captured_but_incomplete = self.island_presence_me.difference(self.island_presence_op).intersection(self.island_presence_ne)
        self.island_complete = set(np.unique(self.island)).difference(self.island_dead).difference(self.island_shared).difference(self.island_captured_but_incomplete)

        self.island_active = self.island_shared.union(self.island_captured_but_incomplete)
        
        self.me_right = self.me_zones[:, :self.me_zones.shape[1] // 2].sum() > self.me_zones[:, self.me_zones.shape[1] // 2:].sum()
        self.kernel_defensive = (np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])if self.me_right else
                                 np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
        # self.kernel_defensive = (np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])if self.me_right else
        #                          np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]))
        # self.kernel_offense = (np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])if self.me_right else
        #                        np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]))


    def set_island_number(self, island_number):
        self.island_number = island_number
        self.island_mask = (self.island == self.island_number)

    def reset_island_number(self):
        self.island_number = -1
        self.island_mask = np.ones((self.height, self.width))

    def cached(self, propery_name, func):
        if propery_name not in self._cached_properties:
            self._cached_properties[propery_name] = {}
        if self.island_number not in self._cached_properties[propery_name]:
            if verbose == 3:
                debug(f'updating cache for island {self.island_number} propery {propery_name}')
            self._cached_properties[propery_name][self.island_number] = func
        return self._cached_properties[propery_name][self.island_number]

    @property
    def me_zones(self):
        if len(np.unique(self.owner_me)) < 2:
            return np.zeros((self.height, self.width))
        row, col = np.where(self.owner > -1)
        X = [(r, c) for r, c in zip(row, col)]
        y = (self.owner[self.owner > -1] == 1).astype(int)
        X_pred = [(row, col) for row in range(self.height) for col in range(self.width)]
        func = LogisticRegression(class_weight='balanced', solver='liblinear').fit(X=X, y=y).predict(X_pred).reshape((self.height, self.width))
        return self.cached('me_zones', func)

    @property
    def op_zones(self):
        if len(np.unique(self.owner_op)) < 2:
            return np.zeros((self.height, self.width))
        row, col = np.where(self.owner > -1)
        X = [(r, c) for r, c in zip(row, col)]
        y = (self.owner[self.owner > -1] == 2).astype(int)
        X_pred = [(row, col) for row in range(self.height) for col in range(self.width)]
        func = LogisticRegression(class_weight='balanced', solver='liblinear').fit(X=X, y=y).predict(X_pred).reshape((self.height, self.width))
        return self.cached('op_zones', func)

    @property
    def my_units_row_distribution(self):
        rows_my_unit_sum = self.units_me.sum(axis=1)
        rows_my_unit_sum_convoled = np.convolve(rows_my_unit_sum, np.ones(3), mode='same')
        func = np.tile(rows_my_unit_sum_convoled, (self.width, 1)).T
        return self.cached('my_units_row_distribution', func)
    
    # @property
    # def my_units_col_distribution(self):
    #     cols_my_unit_sum = self.units_me.sum(axis=0)
    #     cols_my_unit_sum_convoled = np.convolve(cols_my_unit_sum, np.ones(3), mode='same')
    #     func = np.tile(cols_my_unit_sum_convoled, (self.height, 1))
    #     return self.cached('my_units_col_distribution', func)
    
    @property
    def owner_op_col_distribution(self):
        cols_owner_op_sum = self.owner_op.sum(axis=0)
        cols_owner_op_sum_convoled = np.convolve(cols_owner_op_sum, np.ones(5), mode='same')
        func = np.tile(cols_owner_op_sum_convoled, (self.height, 1))
        return self.cached('owner_op_col_distribution', func)

    @property
    def owner_op_col_distribution_global(self):
        func = fftconvolve(self.owner_op_col_distribution * self.island_mask, self.kernel_global, mode='same')
        return self.cached('owner_op_col_distribution_global', func)

    # @property
    # def op_end_global(self):
    #     func = (
    #         np.fliplr(get_kernel_horizonal_gradient(self.height, self.width, 0.95)) if self.me_right else
    #         get_kernel_horizonal_gradient(self.height, self.width, 0.95) * self.island_mask)
    #     return self.cached('op_end_global', func)

    @property
    def my_end_global(self):
        func = (
            get_kernel_horizonal_gradient(self.height, self.width, 0.95) if self.me_right else
            np.fliplr(get_kernel_horizonal_gradient(self.height, self.width, 0.95)) * self.island_mask)
        return self.cached('op_end_global', func)

    @property
    def op_zones_global(self):
        func = fftconvolve(self.op_zones * self.island_mask, self.kernel_global, mode='same')
        return self.cached('op_zones_global', func)

    @property
    def op_zones_manh(self):
        func = np.round(fftconvolve(self.op_zones_global * self.tile_live * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('op_zones_manh', func)

    @property
    def op_zones_manh_manh(self):
        func = np.round(fftconvolve(self.op_zones_manh  * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('op_zones_manh_manh', func)

    # @property
    # def me_zones_global(self):
    #     func = fftconvolve(self.me_zones * self.island_mask, self.kernel_global, mode='same')
    #     return self.cached('me_zones_global', func)

    # @property
    # def me_zones_manh(self):
    #     func = np.round(fftconvolve(self.me_zones_global * self.tile_live * self.island_mask,
    #                                 self.kernel_manhattan, mode='same'), 0).astype(int)
    #     return self.cached('me_zones_manh', func)

    # @property
    # def me_zones_manh_manh(self):
    #     func = np.round(fftconvolve(self.me_zones_manh  * self.island_mask,
    #                                 self.kernel_manhattan, mode='same'), 0).astype(int)
    #     return self.cached('me_zones_manh_manh', func)

    @property
    def live_shared_tile(self):
        func = (self.tile_live * np.isin(self.island, list(self.island_shared))).astype(int)
        return self.cached('live_shared_tile', func)

    @property
    def live_active_tile(self):
        func = (self.tile_live * np.isin(self.island, list(self.island_active))).astype(int)
        return self.cached('live_active_tile', func)

    @property
    def scrap_amount(self):
        func = self._input_array[:, :, 0] * self.island_mask
        return self.cached('scrap_amount', func)

    @property
    def scrap_amount_manh_survivors_n(self):
        func = ((self.scrap_amount > 0) *
            (fftconvolve(self.scrap_amount, np.array([[0,1,0],[0,0,0],[0,0,0]]), mode='same') > self.scrap_amount)
        ).astype(int)
        return self.cached('scrap_amount_manh_survivors_n', func)

    @property
    def scrap_amount_manh_survivors_s(self):
        func = ((self.scrap_amount > 0) *
            (fftconvolve(self.scrap_amount, np.array([[0,0,0],[0,0,0],[0,1,0]]), mode='same') > self.scrap_amount)
        ).astype(int)
        return self.cached('scrap_amount_manh_survivors_s', func)

    @property
    def scrap_amount_manh_survivors_e(self):
        func = ((self.scrap_amount > 0) *
            (fftconvolve(self.scrap_amount, np.array([[0,0,0],[0,0,1],[0,0,0]]), mode='same') > self.scrap_amount)
        ).astype(int)
        return self.cached('scrap_amount_manh_survivors_e', func)

    @property
    def scrap_amount_manh_survivors_w(self):
        func = ((self.scrap_amount > 0) *
            (fftconvolve(self.scrap_amount, np.array([[0,0,0],[1,0,0],[0,0,0]]), mode='same') > self.scrap_amount)
        ).astype(int)
        return self.cached('scrap_amount_manh_survivors_w', func)

    @property
    def scrap_amount_manh_survivors(self):
        func = (self.scrap_amount_manh_survivors_n + self.scrap_amount_manh_survivors_s +
                self.scrap_amount_manh_survivors_e + self.scrap_amount_manh_survivors_w)
        return self.cached('scrap_amount_manh_survivors', func)

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
    def tile_dead_in_t1(self):
        func = self.in_range_of_recycler * (self.scrap_amount == 1) * self.island_mask
        return self.cached('tile_dead_in_t1', func)

    @property
    def can_build(self):
        func = self._input_array[:, :, 4] * self.island_mask
        return self.cached('can_build', func)
    
    # def update_board_with_build(self, row, col):
    #     self._input_array[row, col, 4] = 0
    #     self._input_array[row, col, 5] = 0

    @property
    def can_spawn(self):
        func = self._input_array[:, :, 5] * self.island_mask
        return self.cached('can_spawn', func)

    @property
    def in_range_of_recycler(self):
        func = self._input_array[:, :, 6] * self.island_mask
        return self.cached('in_range_of_recycler', func)

    @property
    def tile_live(self):
        func = ((self.scrap_amount > 0) * (self.recycler == 0) * self.island_mask).astype(int)
        return self.cached('tile_live', func)

    # @property
    # def op_units_in_my_zone(self):
    #     func = self.units_op * self.zones
    #     return self.cached('op_units_in_my_zone', func)

    @property
    def owner_me(self):
        func = ((self.owner == 1) * self.island_mask).astype(int)
        return self.cached('owner_me', func)

    @property
    def owner_op(self):
        func = ((self.owner == 2) * self.island_mask).astype(int)
        return self.cached('owner_op', func)

    @property
    def owner_ne(self):
        func = ((self.owner == -1) * (self.tile_live == 1) * self.island_mask).astype(int)
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
    def in_range_of_my_recycler(self):
        func = np.round(fftconvolve(self.recycler_me * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('in_range_of_my_recycler', func)

    @property
    def recycler_op(self):
        func = self._input_array[:, :, 3] * (self.owner == 2) * self.island_mask
        return self.cached('recycler_me', func)

    # @property
    # def scrap_amount_local(self):
    #     func = np.round(fftconvolve(self.scrap_amount * self.island_mask,
    #                                 self.kernel_manhattan, mode='same'), 0).astype(int)
    #     return self.cached('scrap_amount_local', func)

    @property
    def my_zone_breached_op_unit_positions(self):
        func = (self.me_zones * self.units_op).astype(int)
        return self.cached('my_zone_breached_op_unit_positions', func)

    @property
    def my_zone_breached_deploy_defense(self):
        func = np.round(fftconvolve(self.my_zone_breached_op_unit_positions,
                                    self.kernel_defensive, mode='same'), 0).astype(int)
        return self.cached('my_zone_breached_deploy_defense', func)

    # @property
    # def op_zone_breached_my_owner_positions(self):
    #     func = (self.op_zones * self.owner_me).astype(int)
    #     return self.cached('op_zone_breached_my_owner_positions', func)

    # @property
    # def op_zone_breached_deploy_offense(self):
    #     func = np.round(fftconvolve(self.op_zone_breached_my_owner_positions,
    #                                 self.kernel_offense, mode='same'), 0).astype(int)
    #     return self.cached('op_zone_breached_deploy_offense', func)

    # @property
    # def owner_me_local(self):
    #     func = np.round(fftconvolve(self.owner_me * self.island_mask,
    #                                 self.kernel_manhattan, mode='same'), 0).astype(int)
    #     return self.cached('owner_me_local', func)

    @property
    def owner_op_local(self):
        func = np.round(fftconvolve(self.owner_op * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('owner_op_local', func)

    @property
    def owner_ne_local(self):
        func = np.round(fftconvolve(self.owner_ne * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('owner_ne_local', func)

    # @property
    # def owner_me_global(self):
    #     func = fftconvolve(self.owner_me * self.island_mask, self.kernel_global, mode='same')
    #     return self.cached('owner_me_global', func)

    @property
    def owner_op_global(self):
        func = fftconvolve(self.owner_op * self.island_mask, self.kernel_global, mode='same')
        return self.cached('owner_op_global', func)

    @property
    def owner_ne_global(self):
        func = fftconvolve(self.owner_ne * self.island_mask, self.kernel_global, mode='same')
        return self.cached('owner_ne_global', func)

    # @property
    # def units_op_global(self):
    #     func = fftconvolve(self.units_op * self.island_mask, self.kernel_global, mode='same')
    #     return self.cached('units_op_global', func)

    # @property
    # def units_me_global(self):
    #     func = fftconvolve(self.units_me * self.island_mask, self.kernel_global, mode='same')
    #     return self.cached('units_me_global', func)

    # @property
    # def units_op_local(self):
    #     func = fftconvolve(self.units_op * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('units_op_local', func)

    @property
    def units_me_manh(self):
        func = fftconvolve(self.units_me * self.island_mask, self.kernel_manhattan_vertical, mode='same')
        return self.cached('units_me_manh', func)

    # @property
    # def units_me_manh_2(self):
    #     func = fftconvolve(self.units_me_manh * self.island_mask, self.kernel_manhattan_vertical, mode='same')
    #     return self.cached('units_me_manh_2', func)

    # @property
    # def units_me_manh_3(self):
    #     func = fftconvolve(self.units_me_manh_2 * self.island_mask, self.kernel_manhattan_vertical, mode='same')
    #     return self.cached('units_me_manh_3', func)

    # @property
    # def units_me_manh_4(self):
    #     func = fftconvolve(self.units_me_manh_3 * self.island_mask, self.kernel_manhattan_vertical, mode='same')
    #     return self.cached('units_me_manh_4', func)

    # @property
    # def tile_live_manh_1(self):
    #     func = fftconvolve(self.tile_live * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('tile_live_manh_1', func)

    # @property
    # def tile_live_manh_2(self):
    #     func = fftconvolve(self.tile_live_manh_1 * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('tile_live_manh_2', func)

    # @property
    # def tile_live_manh_3(self):
    #     func = fftconvolve(self.tile_live_manh_2 * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('tile_live_manh_3', func)

    # @property
    # def tile_live_manh_4(self):
    #     func = fftconvolve(self.tile_live_manh_3 * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('tile_live_manh_4', func)

    # @property
    # def tile_live_manh_5(self):
    #     func = fftconvolve(self.tile_live_manh_4 * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('tile_live_manh_5', func)

    # @property
    # def tile_live_manh_6(self):
    #     func = fftconvolve(self.tile_live_manh_5 * self.island_mask, self.kernel_manhattan, mode='same')
    #     return self.cached('tile_live_manh_6', func)

    # @property
    # def units_me_3_3_ones(self):
    #     func = fftconvolve(self.units_me * self.island_mask, self.kernel_3_3_ones, mode='same')
    #     return self.cached('units_me_3_3_ones', func)

def get_move_actions(board, weights, islands):
    actions = []
    for island in islands:
        debug(f'{island=}')
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0, dtype=np.float16)
        scores[board.live_active_tile == False] = np.nan
        scores[board.tile_dead_in_t1 == True] = np.nan
        scores_padded = np.pad(scores, 1, mode='constant', constant_values=np.nan)
        mask = np.array([[np.nan,1,np.nan], [1,np.nan,1], [np.nan,1,np.nan]])
        indices = np.where(board.units_me * board.live_active_tile)
        for row, col in zip(indices[0], indices[1]):
            debug(f'tile {col} {row}')
            scores = scores_padded[row:row+3, col:col+3] * mask
            if np.isnan(scores).all():
                break
            for i in range(1, 1 + board.units_me[row, col]):
                if board.units_me[row, col] > 1:
                    debug(f'unit {i} of {board.units_me[row, col]}')
                debug(f'{np.round(scores, 2)}')
                scores[1, 1] += -10
                row_offset, col_offset = [x - y for x, y in zip(np.unravel_index(np.nanargmax(scores), scores.shape), (1, 1))]
                actions.append(f"MOVE 1 {col} {row} {col + col_offset} {row + row_offset}")
                scores[1+row_offset, 1+col_offset] += -10
    board.reset_island_number()
    return actions

def get_spawn_actions(board, weights, islands, k=1):
    best_action = {'score': -np.inf, 'action': []}
    for island in islands:
        debug(f'{island=}')
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0, dtype=np.float16)
        scores[board.can_spawn == False] = np.nan
        scores[board.live_active_tile == False] = np.nan
        scores[board.tile_dead_in_t1 == True] = np.nan
        if np.isnan(scores).all():
            break
        row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
        verbose and debug(f'island: {island} | score: {np.nanmax(scores)} | action: {f"SPAWN {k} {col} {row}"}')
        if np.nanmax(scores) > best_action['score']:
            best_action = {'score': np.nanmax(scores), 'action': [f"SPAWN {k} {col} {row}"]}
    board.reset_island_number()
    verbose and debug(f'{best_action=}')
    return best_action['action']

def get_build_best_actions(board, weights):
    best_action = {'score': -np.inf, 'action': []}
    islands = list(board.island_active)
    for island in islands:
        debug(f'{island=}')
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0, dtype=np.float16)
        scores[board.can_build == False] = np.nan
        scores[board.live_shared_tile == False] = np.nan
        scores[board.tile_dead_in_t1 == True] = np.nan
        if np.isnan(scores).all():
            break
        row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
        verbose and debug(f'island: {island} | score: {np.nanmax(scores)} | action: {f"BUILD {col} {row}"}')
        if np.nanmax(scores) > best_action['score']:
            best_action = {'score': np.nanmax(scores), 'action': [f"BUILD {col} {row}"]}
    board.reset_island_number()
    verbose and debug(f'{best_action=}')
    return best_action['action']

def get_build_actions(board, weights, k=1):
    actions = []
    islands = list(board.island_active)
    for island in islands:
        debug(f'{island=}')
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0, dtype=np.float16)
        scores[board.can_build == False] = np.nan
        scores[board.live_shared_tile == False] = np.nan
        scores[board.tile_dead_in_t1 == True] = np.nan
        for i in range(k):
            if np.isnan(scores).all():
                break
            row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
            actions.append(f"BUILD {col} {row}")
            scores[row, col] = np.nan
            board.update_board_with_build(row, col)
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
        message = ";".join(action.__str__() for action in actions)
        print(f"{message}")

def main():
    # ======================================================================================
    # Initial Setup
    # ======================================================================================
    debug(f'\n{"="*60}\nInitial Setup\n{"="*60}')
    frame = 0
    io = InputOutput('defensive_breach.txt')
    col, row = io.get_input()
    kernel_global = get_kernel_global(row, col, p=(74 + (2 * (col-12))) / 100)
    perf = []

    while True:
        frame += 1
        try:
            # ======================================================================================
            # Get Inputs
            # ======================================================================================
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
            perf_start = perf_counter()

            # ======================================================================================
            # Board Setup
            # ======================================================================================            
            debug(f'\n{"="*60}\nBoard Setup\n{"="*60}')
            stopclock = perf_counter()
            board = Board(input_array, kernel_global)
            actions = []
            verbose and debug(f'units:     me {board.units_me.sum()} | {board.units_op.sum()} op')
            verbose and debug(f'recyclers: me {board.recycler_me.sum()} | {board.recycler_op.sum()} op')
            verbose and debug(f'owner:     me {board.owner_me.sum()} | {board.owner_op.sum()} op')
            verbose and debug(f'matter:    me {my_matter} | {op_matter} op')
            verbose == 2 and debug(f'\n{board.island}')
            verbose and debug(f'Island Summary:')
            verbose and debug(f' - island_dead\t\t\t\t{board.island_dead}')
            verbose and debug(f' - island_shared\t\t\t{board.island_shared}')
            verbose and debug(f' - island_captured_but_incomplete\t{board.island_captured_but_incomplete}')
            verbose and debug(f' - island_complete\t\t\t{board.island_complete}')
            debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')
            stopclock = perf_counter()

            # ======================================================================================
            # Defensive Zone Breach BUILD / SPAWN actions
            # ======================================================================================
            if True:
                debug(f'\n{"="*60}\nDefensive Zone Breach BUILD / SPAWN actions | matter: {my_matter}\n{"="*60}')
                stopclock = perf_counter()
                verbose == 2 and debug(f'My Zones\n{board.me_zones}')
                verbose == 2 and debug(f'My Zones breached\n{board.my_zone_breached_deploy_defense}')
                if board.my_zone_breached_deploy_defense.sum() == 0:
                    verbose and debug('No zone breaches')
                else:
                    zone_breach_build = board.my_zone_breached_deploy_defense * (board.can_build == 1)
                    zone_breach_spawn = board.my_zone_breached_deploy_defense * (board.can_build == 0) * board.can_spawn
                    build_actions = [f'BUILD {col} {row}' for row, col in zip(*zone_breach_build.nonzero())]
                    spawn_actions = [f'SPAWN 1 {col} {row}' for row, col in zip(*zone_breach_spawn.nonzero())]
                    verbose and debug(f'{board.my_zone_breached_op_unit_positions.sum()} units breached my zone!')
                    verbose and debug(f'{build_actions=}')
                    verbose and debug(f'{spawn_actions=}')
                    zone_breach_actions = build_actions + spawn_actions
                    affordable_zone_breach_actions = zone_breach_actions[:my_matter // 10]
                    verbose and debug(f'{affordable_zone_breach_actions=}')
                    actions += affordable_zone_breach_actions
                    my_matter -= len(affordable_zone_breach_actions) * 10
                debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Maintain minimum recyclers BUILD actions
            # ======================================================================================
            if True:
                THRESHOLD = 0.1
                debug(f'\n{"="*60}\nMaintain minimum recyclers BUILD actions | matter: {my_matter}\n{"="*60}')
                stopclock = perf_counter()
                coverage = board.recycler_me.sum() / (1e-7 + (board.owner_me * board.live_active_tile).sum())
                if my_matter < 10:
                    verbose and debug(f'Not enought matter for BUILD actions')
                elif coverage > THRESHOLD:
                    verbose and debug(f'More than {THRESHOLD} of my tiles are recyclers')
                elif frame < 2:
                    verbose and debug('Do not build on first frame')
                else:
                    verbose and debug(f'{np.round(coverage, 2)} recycler coverage less than {THRESHOLD}... Building 1')
                    new_actions = get_build_best_actions(
                        board=board,
                        weights={
                            'scrap_amount_manh_survivors':  +1,
                            'units_me_manh':                -2,
                            # 'tile_live_manh_3':             -1,
                            'in_range_of_my_recycler':      -1,
                        }
                    )
                    verbose and debug(f'{new_actions=}')
                    actions += new_actions
                    my_matter -= len(new_actions) * 10
                debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Offensive (shared island) MOVE actions
            # ======================================================================================
            if True:
                debug(f'\n{"="*60}\nOffensive (shared island) MOVE actions | matter: {my_matter}\n{"="*60}')
                stopclock = perf_counter()
                # if board.op_zone_breached_my_owner_positions.sum() == 0:
                #     verbose and debug('No zone breaches')
                if len(board.island_shared) == 0:
                    debug(f'No islands shared')
                elif board.units_me.sum() == 0:
                    debug(f'No units to move')
                else:
                    new_actions = get_move_actions(
                        board=board,
                        weights={
                            'my_end_global':                    -1,
                            'owner_op_col_distribution_global': +5,
                            'op_zones_global':                  +1,
                            'op_zones_manh_manh':               +1,
                            'owner_op_global':                  +1,
                            'owner_ne_global':                  +1,
                            # 'tile_live_manh_6':                 +1,
                            'my_units_row_distribution':        -2,
                        },
                        islands=board.island_shared,
                    )
                    verbose and debug(f'{new_actions=}')
                    actions += new_actions
                debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Offensive (shared island) SPAWN actions
            # ======================================================================================
            if True:
                debug(f'\n{"="*60}\nOffensive (shared island) SPAWN actions | matter: {my_matter}\n{"="*60}')
                stopclock = perf_counter()
                verbose == 2 and debug(f'My Zones\n{board.op_zones}')
                verbose == 2 and debug(f'My Zones breached\n{board.op_zone_breached_my_owner_positions}')
                if len(board.island_shared) == 0:
                    debug(f'No islands shared')
                elif board.live_active_tile.sum() == 0:
                    debug(f'No live active tiles to SPAWN to')
                elif my_matter < 10:
                    debug(f'Not enought matter for SPAWN actions')
                else:
                    new_actions = get_spawn_actions(
                        board=board,
                        weights={
                            'my_end_global':                    -1,
                            'owner_op_col_distribution_global': +5,
                            'op_zones_global':                  +1,
                            'op_zones_manh_manh':               +1,
                            'owner_op_global':                  +1,
                            'owner_ne_global':                  +1,
                            # 'tile_live_manh_6':                 +1,
                            'my_units_row_distribution':        -2,
                        },
                        islands=board.island_shared,
                        k=my_matter // 10,
                    )
                    verbose and debug(f'{new_actions=}')
                    actions += new_actions
                    my_matter -= len(new_actions) * 10
                debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Mopup (captured but incomplete island) MOVE actions
            # ======================================================================================
            if True:
                debug(f'\n{"="*60}\nMopup (captured but incomplete island) MOVE actions | matter: {my_matter}\n{"="*60}')
                stopclock = perf_counter()
                if len(board.island_captured_but_incomplete) == 0:
                    debug(f'No islands captured but incomplete')
                elif board.units_me.sum() == 0:
                    debug(f'No units')
                else:
                    new_actions = get_move_actions(
                        board=board,
                        weights={
                            'owner_op_global':   +2,
                            'owner_ne_global':   +1,
                        },
                        islands=board.island_captured_but_incomplete,
                    )
                    verbose and debug(f'{new_actions=}')
                    actions += new_actions
                debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Mopup (captured but incomplete island) SPAWN actions
            # ======================================================================================
            if True:
                debug(f'\n{"="*60}\nMopup (captured but incomplete island) SPAWN actions | matter: {my_matter}\n{"="*60}')
                stopclock = perf_counter()
                if len(board.island_captured_but_incomplete) == 0:
                    debug(f'No islands captured but incomplete')
                elif my_matter < 10:
                    debug(f'Not enought matter for BUILD actions')
                elif board.live_active_tile.sum() == 0:
                    verbose and debug(f'No live active tiles to SPAWN to')
                else:
                    new_actions = get_spawn_actions(
                        board=board,
                        weights={
                            'owner_op_local':   +2,
                            'owner_ne_local':   +1,
                        },
                        islands=board.island_captured_but_incomplete,
                        k=my_matter // 10,
                    )
                    verbose and debug(f'{new_actions=}')
                    actions += new_actions
                    my_matter -= len(new_actions) * 10
                debug(f'perf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Print Actions
            # ======================================================================================
            debug(f'\n{"="*60}\nAction Summary | matter: {my_matter}\n{"="*60}')
            stopclock = perf_counter()
            perf.append(round((perf_counter() - perf_start) * 1000, 1))
            actions += ["WAIT"] if len(actions) == 0 else []
            actions += [f'MESSAGE {perf[-1:][0]}ms ({round(np.mean(perf), 1)}ms)']
            io.send_action(actions)
            debug(f'{int(round((perf_counter() - stopclock) * 1000, 1))} msPrint actions')

            # ======================================================================================
            # Print Logs
            # ======================================================================================
            debug(f'perf:{perf[-1:][0]}ms - Total turn time')


        except EOFError:
            break

if __name__ == '__main__':
    main()
