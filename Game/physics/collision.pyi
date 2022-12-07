from Game.graphic.cartesian import CartesianPlane
from Game.physics.body import object_body


class collision:
    def __init__(self, plane: CartesianPlane) -> None: ...
    def check(self, b1: object_body, b2: object_body) -> None: ...
