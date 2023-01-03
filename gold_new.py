dev = False if __file__ == '/tmp/Answer.py' else True

import sys
import numpy as np
from scipy.signal import fftconvolve
from scipy import ndimage
from time import perf_counter
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    filter[n//2 - 1: n//2 + 2, n//2] = 2
    filter[n//2, n//2 - 1: n//2 + 2] = 2
    return filter


class Board:

    def __init__(self, input_array, p=0.95):
        self._input_array = input_array
        self.height, self.width = input_array.shape[0], input_array.shape[1]
        # self._input_array[:, :, 1] = np.where(self._input_array[:, :, 3] == 1, -1, self._input_array[:, :, 1])
        self._input_array[:, :, 1] = np.where(self._input_array[:, :, 1] == 0, 2, self._input_array[:, :, 1])

        self.kernel_3_3_ones = np.ones((3, 3))
        self.kernel_manhattan = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]])
        self.kernel_global = get_filter(self.height, self.width, p=(74 + (2 * (self.width-12))) / 100)
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
        
        self.island_captured_me = self.island_presence_me.difference(self.island_presence_op)
        self.island_captured_op = self.island_presence_op.difference(self.island_presence_me)
        self.island_captured_ne = self.island_presence_ne.difference(self.island_presence_me).difference(self.island_presence_op)

        self.island_shared = set(self.island_presence_me).intersection(set(self.island_presence_op))
        self.island_captured_but_incomplete = self.island_presence_me.difference(self.island_presence_op).intersection(self.island_presence_ne)
        self.island_complete = set(np.unique(self.island)).difference(self.island_dead).difference(self.island_shared).difference(self.island_captured_but_incomplete)

        self.island_active = self.island_shared.union(self.island_captured_but_incomplete)
        
        self.me_zones = self.get_me_zones()
        self.op_zones = self.get_op_zones()
        self.me_right = self.me_zones[:, :self.me_zones.shape[1] // 2].sum() > self.me_zones[:, self.me_zones.shape[1] // 2:].sum()
        self.kernel_defensive = (np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])if self.me_right else
                                 np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]))
        # self.kernel_offense = (np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])if self.me_right else
        #                        np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]]))
        self.active_island_stats = dict(sorted({
            island: {
                'tiles': np.sum((self.island == island) * self.tile_live),
                'units': {
                    'delta': np.sum((self.island == island) * self.units_me) - np.sum((self.island == island) * self.units_op),
                    'me': np.sum((self.island == island) * self.units_me),
                    'op': np.sum((self.island == island) * self.units_op),
                },
                'owner': {
                    'delta': np.sum((self.island == island) * self.owner_me) - np.sum((self.island == island) * self.owner_op),
                    'me': np.sum((self.island == island) * self.owner_me),
                    'op': np.sum((self.island == island) * self.owner_op),
                },
            }
            for island in self.island_active if island != 0
        }.items(), key=lambda item: item[1]['tiles'], reverse=True))

    def get_me_zones(self):
        if len(np.unique(self.owner_me)) == 1:
            return np.zeros((self.height, self.width))
        df = pd.DataFrame(
            data=[(row, col, self.owner_me[row, col])
                for row in range(self.height)
                for col in range(self.width)],
            columns=['row', 'col', 'owner_me'])
        lr = LogisticRegression(class_weight='balanced', n_jobs=1)
        lr.fit(X=df[['row', 'col']], y=df.owner_me)
        return lr.predict(df[['row', 'col']]).reshape((self.height, self.width))

    def get_op_zones(self):
        if len(np.unique(self.owner_op)) == 1:
            return np.zeros((self.height, self.width))
        df = pd.DataFrame(
            data=[(row, col, self.owner_op[row, col])
                for row in range(self.height)
                for col in range(self.width)],
            columns=['row', 'col', 'owner_op'])
        lr = LogisticRegression(class_weight='balanced', n_jobs=1)
        lr.fit(X=df[['row', 'col']], y=df.owner_op) 
        return lr.predict(df[['row', 'col']]).reshape((self.height, self.width))

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
            # if verbose:
            #     debug(f'updating cache for island {self.island_number} propery {propery_name}')
            self._cached_properties[propery_name][self.island_number] = func
        return self._cached_properties[propery_name][self.island_number]

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
    
    def update_board_with_build(self, row, col):
        self._input_array[row, col, 4] = 0
        self._input_array[row, col, 5] = 0

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

    @property
    def op_units_in_my_zone(self):
        func = self.units_op * self.zones
        return self.cached('op_units_in_my_zone', func)

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
    def scrap_amount_local(self):
        func = np.round(fftconvolve(self.scrap_amount * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('scrap_amount_local', func)

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
    # def op_zone_breached_my_unit_positions(self):
    #     func = (self.op_zones * self.units_me).astype(int)
    #     return self.cached('op_zone_breached_my_unit_positions', func)

    # @property
    # def op_zone_breached_deploy_offense(self):
    #     func = np.round(fftconvolve(self.op_zone_breached_my_unit_positions,
    #                                 self.kernel_offense, mode='same'), 0).astype(int)
    #     return self.cached('op_zone_breached_deploy_offense', func)

    @property
    def owner_me_local(self):
        func = np.round(fftconvolve(self.owner_me * self.island_mask,
                                    self.kernel_manhattan, mode='same'), 0).astype(int)
        return self.cached('owner_me_local', func)

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

def get_move_actions_new(board, weights, islands):
    actions = []
    for island in islands:
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
        scores[board.live_active_tile == False] = np.nan
        scores_padded = np.pad(scores, 1, mode='constant', constant_values=np.nan)
        mask = np.array([[np.nan,1,np.nan], [1,np.nan,1], [np.nan,1,np.nan]])
        indices = np.where(board.units_me * board.live_active_tile)
        for row, col in zip(indices[0], indices[1]):
            scores = scores_padded[row:row+3, col:col+3] * mask
            for i in range(1, 1 + board.units_me[row, col]):
                scores[1, 1] = -1e7
                row_offset, col_offset = [x - y for x, y in zip(np.unravel_index(np.nanargmax(scores), scores.shape), (1, 1))]
                actions.append(f"MOVE 1 {col} {row} {col + col_offset} {row + row_offset}")
                scores[1+row_offset, 1+col_offset] = np.nan
    board.reset_island_number()
    return actions

def get_spawn_actions_new(board, weights, islands, k=1):
    best_action = {'score': -np.inf, 'action': []}
    for island in islands:
        board.set_island_number(island)
        scores = np.nansum([weight * min_max(getattr(board, feature)) for feature, weight in weights.items()], axis=0)
        scores[board.can_spawn == False] = np.nan
        scores[board.live_active_tile == False] = np.nan
        if np.isnan(scores).all():
            break
        row, col = np.unravel_index(np.nanargmax(scores), scores.shape)
        verbose and debug(f'\tisland: {island} | score: {np.nanmax(scores)} | action: {f"SPAWN {k} {col} {row}"}')
        if np.nanmax(scores) > best_action['score']:
            best_action = {'score': np.nanmax(scores), 'action': [f"SPAWN {k} {col} {row}"]}
    board.reset_island_number()
    verbose and debug(f'\t{best_action=}')
    return best_action['action']

def get_spawn_actions(board, weights, k=1):
    actions = []
    islands = list(board.island_active)
    for island in islands:
        board.set_island_number(island)
        new_spawns = np.zeros((board.height, board.width), dtype=np.float16)
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
    frame = 0
    io = InputOutput('p272_crash.txt')
    col, row = io.get_input()
    perf = []

    while True:
        frame += 1
        perf_start = perf_counter()
        try:
            # ======================================================================================
            # Get Inputs
            # ======================================================================================
            verbose and debug(f'\n{"="*40}\nGet Inputs\n{"="*40}')
            stopclock = perf_counter()
            my_matter, op_matter = io.get_input()
            input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
            debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 0))} ms')

            # ======================================================================================
            # Board Setup
            # ======================================================================================            
            verbose and debug(f'\n{"="*40}\nIslands\n{"="*40}')
            stopclock = perf_counter()
            board = Board(input_array)
            actions = []
            verbose == 2 and debug(f'{board.island}')
            verbose and debug(f'\tisland_dead\t\t\t{board.island_dead}')
            verbose and debug(f'\tisland_shared\t\t\t{board.island_shared}')
            verbose and debug(f'\tisland_captured_but_incomplete\t{board.island_captured_but_incomplete}')
            verbose and debug(f'\tisland_complete\t\t\t{board.island_complete}')
            debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')
            stopclock = perf_counter()

            # ======================================================================================
            # Defensive Zone Breach actions
            # ======================================================================================
            verbose and debug(f'\n{"="*40}\nDefensive Zone Breach Actions ({my_matter})\n{"="*40}')
            stopclock = perf_counter()
            verbose == 2 and debug(f'My Zones\n{board.me_zones}')
            verbose == 2 and debug(f'My Zones breached\n{board.my_zone_breached_op_unit_positions}')
            if board.my_zone_breached_op_unit_positions.sum() == 0:
                verbose and debug('\tNo zone breaches')
            else:
                zone_breach_build = board.my_zone_breached_deploy_defense * (board.can_build == 1)
                zone_breach_spawn = board.my_zone_breached_deploy_defense * (board.can_build == 0) * board.can_spawn
                build_actions = [f'BUILD {col} {row}' for row, col in zip(*zone_breach_build.nonzero())]
                spawn_actions = [f'SPAWN 1 {col} {row}' for row, col in zip(*zone_breach_spawn.nonzero())]
                verbose and debug(f'\t{board.my_zone_breached_op_unit_positions.sum()} units breached my zone!')
                verbose and debug(f'\t{build_actions=}')
                verbose and debug(f'\t{spawn_actions=}')
                zone_breach_actions = build_actions + spawn_actions
                affordable_zone_breach_actions = zone_breach_actions[:my_matter // 10]
                verbose and debug(f'\t{affordable_zone_breach_actions=}')
                actions += affordable_zone_breach_actions
                my_matter -= len(affordable_zone_breach_actions) * 10
            debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')

            # ======================================================================================
            # Offensive Zone Breach actions
            # ======================================================================================
            verbose and debug(f'\n{"="*40}\nOffensive Zone Breach Actions ({my_matter})\n{"="*40}')
            stopclock = perf_counter()
            verbose == 2 and debug(f'Op Zones\n{board.op_zones}')
            verbose and debug(f'\tOp Zones breached by me - TBC')
            if board.my_zone_breached_op_unit_positions.sum() == 0:
                verbose and debug('\tNo zone breaches')
            else:
                # offensive SPAWN actions
                pass
            debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')


            # ======================================================================================
            # BUILD actions
            # ======================================================================================
            verbose and debug(f'\tBUILD actions ({my_matter})\n\t{"-"*40}')
            stopclock = perf_counter()
            coverage = board.recycler_me.sum() / (1e-7 + (board.owner_me * board.live_active_tile).sum())
            if my_matter < 10:
                verbose and debug(f'\tNot enought matter for BUILD actions')
            elif coverage > .15:
                verbose and debug(f'\tMore than 15% of my tiles are recyclers')
            elif frame < 3:
                verbose and debug('Do not build on first two frames')
            else:
                verbose and debug(f'\t{np.round(coverage, 2)} recycler coverage... less than 15%')
                new_actions = get_build_actions(
                    board=board,
                    weights={
                        'units_op_global':  +1,
                    }
                )
                verbose and debug(f'\t{new_actions=}')
                actions += new_actions
                my_matter -= len(new_actions) * 10
            debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')


            # ======================================================================================
            # Island (Shared) actions
            # ======================================================================================
            verbose and debug(f'\n{"="*40}\nIsland (Shared) Actions ({my_matter})\n{"="*40}')
            if len(board.island_shared) == 0:
                debug(f'\tNo islands shared')
            else:
                # MOVE actions
                verbose and debug(f'\tMOVE actions ({my_matter})\n\t{"-"*20}')
                stopclock = perf_counter()
                if board.units_me.sum() > 0:
                    new_actions = get_move_actions_new(
                        board=board,
                        weights={
                            'op_end_global':    +12,
                            'owner_op_global':  +5,
                            'units_me_global':  -6,
                        },
                        islands=board.island_shared,
                    )
                    verbose and debug(f'\t{new_actions=}')
                    actions += new_actions
                debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')
                
                # SPAWN actions
                verbose and debug(f'\tSPAWN actions ({my_matter})\n\t{"-"*20}')
                if my_matter < 10:
                    debug(f'\tNot enought matter for SPAWN or BUILD actions')
                else:
                    stopclock = perf_counter()
                    if my_matter < 10:
                        verbose and debug(f'\tNot enought matter for BUILD actions')
                    elif board.live_active_tile.sum() == 0:
                        verbose and debug(f'\tNo live active tiles to SPAWN to')
                    else:
                        new_actions = get_spawn_actions_new(
                            board=board,
                            weights={
                                'op_end_global':    +12,
                                'owner_op_global':  +5,
                                'units_me_global':  -6,
                                # 'owner_op_local':   +4,
                                # 'owner_ne_local':   +3,
                            },
                            islands=board.island_shared,
                            k=my_matter // 10,
                        )
                        verbose and debug(f'\t{new_actions=}')
                        actions += new_actions
                        my_matter -= len(new_actions) * 10
                    debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')


            # ======================================================================================
            # Island (captured but incomplete) actions
            # ======================================================================================
            verbose and debug(f'\n{"="*40}\nIsland (captured but incomplete) Actions ({my_matter})\n{"="*40}')
            if len(board.island_captured_but_incomplete) == 0:
                debug(f'\tNo islands captured but incomplete')
            else:
                # MOVE actions
                verbose and debug(f'\tMOVE actions ({my_matter})\n\t{"-"*20}')
                stopclock = perf_counter()
                if board.units_me.sum() > 0:
                    new_actions = get_move_actions_new(
                        board=board,
                        weights={
                            'owner_op_global':   +2,
                            'owner_ne_global':   +1,
                        },
                        islands=board.island_captured_but_incomplete,
                    )
                    verbose and debug(f'{new_actions=}')
                    actions += new_actions
                debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')
                
                # SPAWN actions
                verbose and debug(f'\tSPAWN actions ({my_matter})\n\t{"-"*20}')
                if my_matter < 10:
                    debug(f'\tNot enought matter for SPAWN or BUILD actions')
                else:
                    verbose and debug(f'\tSPAWN actions ({my_matter})\n{"-"*20}')
                    stopclock = perf_counter()
                    if my_matter < 10:
                        verbose and debug(f'\tNot enought matter for BUILD actions')
                    elif board.live_active_tile.sum() == 0:
                        verbose and debug(f'\tNo live active tiles to SPAWN to')
                    else:
                        new_actions = get_spawn_actions_new(
                            board=board,
                            weights={
                                'owner_op_local':   +2,
                                'owner_ne_local':   +1,
                            },
                            islands=board.island_captured_but_incomplete,
                            k=my_matter // 10,
                        )
                        verbose and debug(f'\t{new_actions=}')
                        actions += new_actions
                        my_matter -= len(new_actions) * 10
                    debug(f'\tperf: {int(round((perf_counter() - stopclock) * 1000, 1))} ms')
            


            # # get MOVE actions
            # stopclock = perf_counter()
            # verbose and debug(f'\nMOVE actions ({my_matter})\n{"-"*40}')
            # if board.units_me.sum() > 0:
            #     move_actions = get_move_actions(board, move_weights)
            #     verbose and debug(f'{move_actions=}')
            #     actions += move_actions
            # debug(f'{int(round((perf_counter() - stopclock) * 1000, 1))} ms\tGet MOVE actions')


            # # get BUILD actions
            # stopclock = perf_counter()
            # verbose and debug(f'\nBUILD actions ({my_matter})\n{"-"*40}')
            # if my_matter < 10:
            #     verbose and debug(f'Not enought matter for BUILD actions')
            # elif board.recycler_me.sum() / (1e-7 + (board.owner_me * board.live_active_tile).sum()) > .25:
            #     verbose and debug(f'More than 15% of my tiles are recyclers')
            # else:
            #     build_actions = get_build_actions(board, build_weights)
            #     verbose and debug(f'{build_actions=}')
            #     actions += build_actions
            #     my_matter -= len(build_actions) * 10
            # debug(f'{int(round((perf_counter() - stopclock) * 1000, 1))} ms\tGet BUILD actions')


            # # get SPAWN actions
            # stopclock = perf_counter()
            # verbose and debug(f'\nSPAWN actions ({my_matter})\n{"-"*40}')
            # if my_matter < 10:
            #     verbose and debug(f'Not enought matter for BUILD actions')
            # elif board.live_active_tile.sum() == 0:
            #     verbose and debug(f'No live active tiles to SPAWN to')
            # else:
            #     spawn_actions = get_spawn_actions(board, spawn_weights, k=my_matter // 10)
            #     verbose and debug(f'{spawn_actions=}')
            #     actions += spawn_actions
            #     my_matter -= len(spawn_actions) * 10
            # debug(f'{int(round((perf_counter() - stopclock) * 1000, 1))} ms\tGet SPAWN actions')

            # ======================================================================================
            # Print Actions
            # ======================================================================================
            verbose and debug(f'\n{"="*40}\nFinal Actions ({my_matter})\n{"="*40}')
            stopclock = perf_counter()
            perf.append(round((perf_counter() - perf_start) * 1000, 1))
            actions += ["WAIT"] if len(actions) == 0 else []
            actions += [f'MESSAGE {perf[-1:][0]}ms ({round(np.mean(perf), 1)}ms)']
            io.send_action(actions)
            debug(f'{int(round((perf_counter() - stopclock) * 1000, 1))} ms\tPrint actions')

            # ======================================================================================
            # Print Logs
            # ======================================================================================
            debug(f'perf:{perf[-1:][0]}ms - Total turn time')


        except EOFError:
            break

if __name__ == '__main__':
    main()
