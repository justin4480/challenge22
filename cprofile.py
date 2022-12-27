import cProfile
from challenge22 import main

cProfile.run('main()', sort='tottime')
