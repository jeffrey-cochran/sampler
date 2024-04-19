from enum import IntFlag, auto

class BoundaryFlag(IntFlag):
    BOTTOM = auto()
    LEFT = auto()
    BACK = auto()
    TOP = auto()
    RIGHT = auto()
    FRONT = auto()
    X = LEFT | RIGHT
    Y = FRONT | BACK
    Z = TOP | BOTTOM

DIM = {
    BoundaryFlag.LEFT: 0,
    BoundaryFlag.RIGHT: 0,
    BoundaryFlag.BACK: 1,
    BoundaryFlag.FRONT: 1,
    BoundaryFlag.TOP: 2,
    BoundaryFlag.BOTTOM: 2
}

GRAVITY = 9.8