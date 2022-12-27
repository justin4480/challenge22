import pytest
import numpy as np
from challenge22 import InputOutput, Board


@pytest.fixture
def board():
    io = InputOutput('unittest.txt')
    col, row = io.get_input()
    my_matter, op_matter = io.get_input()
    input_array = np.array(io.get_input(row * col)).reshape(row, col, 7)
    board = Board(input_array)
    return board


def test_array_shape(board):
    assert board._input_array.shape == (6, 12, 7)





# @pytest.fixture
# def hdd_values():
#     return {
#         'name': '1TB SATA HDD',
#         'manufacturer': 'Seagate',
#         'total': 10,
#         'allocated': 3,
#         'capacity_gb': 1_000,
#         'size': '3.5"',
#         'rpm': 10_000
#     }


# @pytest.fixture
# def hdd(hdd_values):
#     return inventory.HDD(**hdd_values)


# def test_create(hdd, hdd_values):
#     for attr_name in hdd_values:
#         assert getattr(hdd, attr_name) == hdd_values.get(attr_name)


# @pytest.mark.parametrize('size', ['2.5', '5.25"'])
# def test_create_invalid_size(size, hdd_values):
#     hdd_values['size'] = size
#     with pytest.raises(ValueError):
#         inventory.HDD(**hdd_values)