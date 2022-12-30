import cProfile
from gold_700 import main

# cProfile.run('main()', sort='tottime')
cProfile.run('main()', sort='cumtime')

# 2113 function calls (2059 primitive calls) in 0.013 seconds p329
# 2104 function calls (2050 primitive calls) in 0.024 seconds challenge