import sys

def debug(message) -> None:
    print(message, file=sys.stderr, flush=True)

while True:
    debug(input())
    print('WAIT')
