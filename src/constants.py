from enum import IntFlag, auto

class BoundaryFlag(IntFlag):
    BOTTOM = auto()
    LEFT = auto()
    TOP = auto()
    RIGHT = auto()
    X = LEFT | RIGHT
    Y = TOP | BOTTOM

DIM = {
    BoundaryFlag.LEFT: 0,
    BoundaryFlag.RIGHT: 0,
    BoundaryFlag.TOP: 1,
    BoundaryFlag.BOTTOM: 1
}

GRAVITY = 9.8