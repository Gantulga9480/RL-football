from Game.graphic.cartesian import CartesianPlane


class EnginePolygon:

    def __init__(self, plane: CartesianPlane, bodies: memoryview) -> None: ...
    def step(self) -> None: ...
