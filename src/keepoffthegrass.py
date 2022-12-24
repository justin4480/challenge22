dev = False if __file__ == '/tmp/Answer.py' else True

import sys
import numpy as np
from scipy.signal import convolve2d
from time import perf_counter
from abc import ABC, abstractmethod
if dev:
    from src.fake_input import input_stream

actionid = 0
VERBOSE = 0
VALIDATION = 0

def debug(message) -> None:
    print(message, file=sys.stderr, flush=True)


class ActionLogger:
    def __init__(self, board):
        self.actions = []
        self.board = board

    def add(self, action):
        VALIDATION and self.board.validate_proposed_action(action)
        global actionid
        actionid += 1
        debug(f"{actionid} Action {action.__str__()}")
        debug(f"\tB: Matter: {self.board.my_matter_amount} | Units: {len(self.board.available_units)}")
        VERBOSE and debug(f"{self.board}")
        VERBOSE and debug(f"action: {action}")
        self.actions.append(action)
        self.board.new_action_board_refresh(action)
        self.board.new_action_spend(action)
        VERBOSE and debug(f"{self.board}\n")
        debug(f"\tA: Matter: {self.board.my_matter_amount} | Units: {len(self.board.available_units)}")


class BuildAction:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"BUILD {self.x} {self.y}"


class SpawnAction:
    def __init__(self, amount, x, y):
        self.amount = amount
        self.x = x
        self.y = y

    def __str__(self):
        return f"SPAWN {self.amount} {self.x} {self.y}"


class MoveAction:
    def __init__(self, from_x, from_y, to_x, to_y, amount=1):
        self.amount = amount
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.creater_function = None

    def __str__(self):
        return f"MOVE {self.amount} {self.from_x} {self.from_y} {self.to_x} {self.to_y}"


class Board:

    FEATURES = {
        "scrap_amount": 0,
        "owner": 1,
        "units": 2,
        "recycler": 3,
        "can_build": 4,
        "can_spawn": 5,
        "in_range_of_recycler": 6,
        "can_move": 7,
        "occupiable_tile": 8,
    }

    def __init__(self, x, y, raw_board_input, my_matter_amount, op_matter_amount):
        self.x = x
        self.y = y
        self._array = np.array(raw_board_input).reshape(self.row, self.col, 7)
        self.add_can_move_feature()
        self.my_matter_amount = my_matter_amount
        self.op_matter_amount = op_matter_amount
    
    @property
    def available_units(self):
        return self.int_array_to_xy(self.dim_to_array('can_move') == 1)
    
    @property
    def my_units_xy(self):
        return self.int_array_to_xy(self.dim_to_array('units') * (self.dim_to_array('owner') == 1))
    
    @property
    def op_units_xy(self):
        return self.int_array_to_xy(self.dim_to_array('units') * (self.dim_to_array('owner') == 0))

    def add_features(self):
        # can_move
        can_move = self.dim_to_array('units') * (self.dim_to_array('owner') == 1)
        self._array = np.concatenate((self._array, can_move[:,:,None]), axis=2)

        # occupiable_tile
        occupiable_tile = (self.dim_to_array('scrap_amount') > 0) * (self.dim_to_array('recycler') == 0)
        self._array = np.concatenate((self._array, occupiable_tile[:,:,None]), axis=2)

    def validate_dimension_name(self, dimension):
        if dimension not in Board.FEATURES.keys():
            raise ValueError(f"{dimension} is not in {Board.FEATURES.keys()}")

    def validate_proposed_action(self, action) -> bool:
        if isinstance(action, MoveAction):
            if self._array[:,:,self.FEATURES['can_move']][action.from_y, action.from_x] != 1:
                raise ValueError(f"can_move Validation Failed")
        if isinstance(action, BuildAction):
            if self._array[:,:,self.FEATURES['can_build']][action.y, action.x] != 1:
                raise ValueError(f"can_build Validation Failed")
            if self.my_matter_amount < 10:
                raise ValueError(f"sufficient matter Validation Failed")
        if isinstance(action, SpawnAction):
            if self._array[:,:,self.FEATURES['can_spawn']][action.y, action.x] != 1:
                raise ValueError(f"can_spawn Validation Failed")
            if self.my_matter_amount < 10:
                raise ValueError(f"sufficient matter Validation Failed")

    def new_action_spend(self, action):
        if isinstance(action, MoveAction):
            self._array[action.from_y, action.from_x, self.FEATURES['can_move']] = 0
        if isinstance(action, BuildAction):
            self.my_matter_amount -= 10
        if isinstance(action, SpawnAction):
            assert self._array[action.y, action.x, self.FEATURES['can_move']] == 0
            # self._array[action.y, action.x, self.FEATURES['can_move']] = 0  # cannot move after spawning
            self.my_matter_amount -= (10 * action.amount)

    def new_action_board_refresh(self, action):
        if isinstance(action, MoveAction):
            self._array[:,:,self.FEATURES['units']][action.from_y, action.from_x] -= 1
            self._array[:,:,self.FEATURES['units']][action.to_y, action.to_x] += 1
        if isinstance(action, BuildAction):
            self._array[:,:,self.FEATURES['recycler']][action.y, action.x] += 1
        if isinstance(action, SpawnAction):
            self._array[:,:,self.FEATURES['units']][action.y, action.x] += action.amount

    def __str__(self):
        return (
            f"{'='*30}\n"
            f"Matter: {self.my_matter_amount} | Units: {len(self.available_units)} {self.my_units_xy}"
            f"\n{'='*30}"
        )

    @property
    def row(self) -> int:
        """used for numpy array indexing"""
        return self.y

    @property
    def col(self) -> int:
        """used for numpy array indexing"""
        return self.x

    @staticmethod
    def validate_kernel(kernel: np.ndarray):
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("kernel must be odd in both dimensions")

    @staticmethod
    def normalise(array: np.ndarray) -> np.ndarray:
        return (array - np.mean(array)) / np.std(array)

    def execute_actions(self, actions: list):
        pass

    def dim_to_array(self, dimension: str, normalised: bool = False) -> np.ndarray:
        """example: dim_to_array('units') == -1"""
        self.validate_dimension_name(dimension)
        array = self._array[:, :, self.FEATURES[dimension]]
        return self.normalise(array) if normalised else array

    def dim_to_conv(self, dimension: str, kernel: np.ndarray = np.ones((3, 3)),
                    normalise: bool = False) -> np.ndarray:
        """example: dim_to_conv('matter', True)"""
        self.validate_dimension_name(dimension)
        array = self.dim_to_array(dimension)
        array_convolved = self.array_to_conv(array, kernel, normalise)
        return self.normalise(array_convolved) if normalise else array_convolved

    def convolve2d(self, array, kernel):
        return convolve2d(array, kernel, mode='same')

    def array_to_conv(self, array: np.ndarray, kernel: np.ndarray = np.ones((3, 3)),
                      normalise: bool = False) -> np.ndarray:
        """example:
            enemy_units = board.dim_to_array('owner') == 1 * board.dim_to_array('units') > 0
            array_to_conv(enemy_units)
        """
        self.validate_kernel(kernel)
        array = self.convolve2d(array, kernel)
        return self.normalise(array) if normalise else array

    def int_array_to_xy(self, array: np.ndarray) -> list:
        """returns the x,y coordinates of all values in the array; accounting for duplicates"""
        xy = []
        for i in range(1, array.max() + 1):
            xy += [(j[0], j[1]) for j in np.argwhere(array == i)[:, ::-1]] * i
        return xy

    def bool_array_to_xy(self, array: np.ndarray) -> list:
        """returns the x,y coordinates of all 1's in the array"""
        if array.max() > 1:
            raise ValueError("array must only contain 0's and 1's")
        return [(i[0], i[1]) for i in np.argwhere(array == 1)[:, ::-1]]

    def top_n_positions(self, array: np.ndarray, k: int = 1, descending=True):
        np.random.seed(0)
        tiebreaker = np.random.rand(*array.shape)
        array = array + tiebreaker
        sorted_arr = np.sort(array, axis=None, kind='quicksort', order=None)
        if descending:
            sorted_dec_arr = sorted_arr[-k:][::-1]
            return [tuple(np.argwhere(array == i)[0][::-1]) for i in sorted_dec_arr][:k]
        else:
            sorted_asc_arr = sorted_arr[:k]
            return [tuple(np.argwhere(array == i)[0][::-1]) for i in sorted_asc_arr][:k]

    def pad_array_to_odd_dims(self, array: np.ndarray) -> np.ndarray:
        """pads array with zeros to make dimensions odd numbers"""
        row, col = array.shape
        if row % 2 == 0:
            array = np.pad(array, ((0, 1), (0, 0)), mode='constant')
        if col % 2 == 0:
            array = np.pad(array, ((0, 0), (0, 1)), mode='constant')
        return array

    @property
    def heatmap_center(self) -> np.ndarray:
        # pad to make dimensions odd numbers
        padded_row = self.row - 1 if self.row % 2 == 0 else self.row
        padded_col = self.col - 1 if self.col % 2 == 0 else self.col
        return self.array_to_conv(
            array=np.ones((self.row, self.col)),
            kernel=np.ones((padded_row, padded_col))
        )


# class Strategy:
#     def __init__(self, action_logger, inventory):
#         self.action_logger = action_logger
#         self.inventory = inventory

#     @abstractmethod
#     def process_events(self):
#         pass

#     @property
#     def print_actions(self):
#         return self._print_actions


class Array:
    def get_subarray(array, row, col):
        return array[max(0, row-1):row+2, max(0, col-1):col+2]


class Distance:

    @staticmethod
    def get_manhattan_distance(xy1, xy2):
        x1, y1 = xy1
        x2, y2 = xy2
        return abs(x1 - x2) + abs(y1 - y2)
    
    # @staticmethod
    # def get_single_manhattan_step(xy1, xy2, constrained_to=None):
    #     x1, y1 = xy1
    #     x2, y2 = xy2


    @staticmethod
    def get_single_manhattan_step(xy1, xy2):
        x1, y1 = xy1
        x2, y2 = xy2
        if abs(x1 - x2) > abs(y1 - y2):
            x1 += 1 if (x2 - x1) > 0 else -1
        else:
            y1 += 1 if (y2 - y1) > 0 else -1
        return (x1, y1)

    @staticmethod
    def get_closest(xys: list[tuple[int, int]], xy: tuple[int, int]):
        closest_distance = 1e10
        closest_xy = ()
        for xy_candidate in xys:
            distance = Distance.get_manhattan_distance(xy_candidate, xy)
            if distance < closest_distance:
                closest_xy = xy_candidate
                closest_distance = distance
        return closest_xy


class Strategy:

    def __init__(self, action_logger, board):
        self.action_logger = action_logger
        self.board = board

    def process_events(self):
        self.move_events()
        self.build_events()
        self.spawn_events()
    
    def build_events(self):
        if self.board.my_matter_amount >= 10:
            array = (
                1
                * (self.board.dim_to_conv('scrap_amount'))
                * (self.board.dim_to_array('in_range_of_recycler') == 0)
                * (self.board.dim_to_array('can_build') == 1)
            )
            x, y = self.board.top_n_positions(array=array, k=1)[0]
            self.action_logger.add(BuildAction(x, y))
    
    def spawn_events(self):
        while self.board.my_matter_amount >= 10:
            array = (
                1
                * (self.board.dim_to_array('scrap_amount'))
                * (self.board.dim_to_array('units') == 0)
                * (self.board.dim_to_array('can_spawn') == 1)
            )
            x, y = self.board.top_n_positions(array=array, k=1)[0]
            self.action_logger.add(SpawnAction(1, x, y))

    def move_events(self):

        assert np.all((self.board.dim_to_array('can_move') == 1) == (
            1
            * self.board.dim_to_array('owner') == 1)
            * (self.board.dim_to_array('units') == 1)
            * (self.board.dim_to_array('can_move') == 1)
        )

        # move to best unavailable build position
        if False:
            if len(self.board.available_units) > 0:
                array = (
                    1
                    * (self.board.dim_to_conv('scrap_amount'))
                    * (self.board.dim_to_array('in_range_of_recycler') == 0)
                    * (self.board.dim_to_array('can_build') == 0)
                )
                top_position = self.board.top_n_positions(array=array, k=1)[0]
                from_xy = Distance.get_closest(xys=self.board.available_units, xy=top_position)
                to_xy = Distance.get_single_manhattan_step(xy1=from_xy, xy2=top_position)
                self.action_logger.add(MoveAction(*from_xy, *to_xy))

        # move to unowned
        if True:
            while len(self.board.available_units) > 0:
                # array = self.board.array_to_conv(self.board.dim_to_array('owner') == -1, np.ones((13, 13)))
                array = self.board.array_to_conv(self.board.dim_to_array('owner') != 1, np.ones((3, 3)))
                top_position = self.board.top_n_positions(array=array, k=1)[0]                
                from_xy = Distance.get_closest(xys=self.board.available_units, xy=top_position)
                
                self.board.dim_to_array('can_move')
                Array.get_subarray(from_xy)

                to_xy = Distance.get_single_manhattan_step(xy1=from_xy, xy2=top_position)
                self.action_logger.add(MoveAction(*from_xy, *to_xy))

        # move to center
        if False:
            if len(self.board.available_units) > 0:
                to_x, to_y = self.board.top_n_positions(array=self.board.heatmap_center, k=1)[0]
                # from_x, from_y = self.board.available_units.pop()
                self.action_logger.add(MoveAction(from_x, from_y, to_x, to_y))

        # move remaining units towards enemy units
        if True:
            while len(self.board.available_units) > 0:
                my_units_conv = self.board.dim_to_array('units') * (self.board.dim_to_array('owner') == 1)
                op_units_conv = self.board.dim_to_array('units') * (self.board.dim_to_array('owner') == 0)
                dominating = my_units_conv > self.board.array_to_conv(op_units_conv)
                top_position = self.board.top_n_positions(dominating)[0]
                to_xy = Distance.get_single_manhattan_step(xy1=from_xy, xy2=top_position)
                self.action_logger.add(MoveAction(*from_xy, *to_xy))
        
        # safty in numbers strategy


class InputOutput:

    def __init__(self, filename: str = "unittest.txt"):
        self.input = input_stream(filename) if dev else input

    def get_input(self, n: int = 1) -> None:
        lst = [self.input().split() for _ in range(n)]
        return [int(item) for sublist in lst for item in sublist]

    def send_action(self, actions: list):
        message = ";".join(action.__str__() for action in actions)
        print(f"{message}")


def main():
    io = InputOutput('right.txt')
    x, y = io.get_input()
    frame = 0

    while True:
        start = perf_counter()
        frame += 1
        debug(f"\n{'='*50}\nframe = {frame}\n{'='*50}")
        try:
            # get inputs
            my_matter_amount, op_matter_amount = io.get_input()
            board = Board(x, y, io.get_input(x * y), my_matter_amount, op_matter_amount)

            # reset objects
            action_logger = ActionLogger(board)
            strategy = Strategy(action_logger, board)

            # processing
            strategy.process_events()

            # execute actions
            VERBOSE and debug(f"{'-'*50}")
            io.send_action(action_logger.actions)
            VERBOSE and debug(f"{'-'*50}")
            stop = perf_counter()
            debug(f"Time: {stop - start:.2f}s")
        except EOFError:
            if dev:
                debug("End of input stream")
            break

if not dev:
    main()
