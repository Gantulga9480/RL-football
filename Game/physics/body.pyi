from Game.graphic.cartesian import CartesianPlane, Vector2d
from Game.graphic.shapes import Shape


class Body:
    type: int
    id: int
    radius: float
    friction_coef: float
    drag_coef: float
    is_attached: bool
    is_following_dir: bool
    shape: Shape
    velocity: Vector2d
    def __init__(self, id: int, type: int) -> None: ...

    def position(self) -> tuple:
        """
        @return
        (x, y) Coordinate in Cartesian space.
        """
        ...

    def direction(self) -> float:
        """
        @return
        In Degrees [0 to 360)
        """
        ...

    def speed(self) -> float: ...
    def step(self) -> None: ...
    def attach(self, o: Body, follow_dir: bool) -> None: ...
    def detach(self) -> None: ...
    def rotate(self, angle: float) -> None: ...
    def scale(self, factor: float) -> None: ...
    def show(self, vertex: bool = False, velocity: bool = False, width: int = 1) -> None: ...


class FreeBody(Body):
    def __init__(self, id: int, plane: CartesianPlane, max_speed: float = 0, drag_coef: float = 0) -> None: ...
    def accelerate(self, speed: float) -> None: ...


class StaticBody(Body):
    def __init__(self, id: int, plane: CartesianPlane) -> None: ...


class DynamicBody(Body):
    def __init__(self, id: int, plane: CartesianPlane, max_speed: float = 1, drag_coef: float = 0.03, friction_coef: float = 0.3) -> None: ...
    def accelerate(self, speed: float) -> None: ...


class DynamicPolygonBody(DynamicBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 1, drag_coef: float = 0.03, friction_coef: float = 0.3) -> None: ...


class DynamicRectangleBody(DynamicBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 1, drag_coef: float = 0.03, friction_coef: float = 0.3) -> None: ...


class DynamicTriangleBody(DynamicBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 1, drag_coef: float = 0.03, friction_coef: float = 0.3) -> None: ...


class StaticPolygonBody(StaticBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple) -> None: ...


class StaticRectangleBody(StaticBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple) -> None: ...


class StaticTriangleBody(StaticBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple) -> None: ...


class FreePolygonBody(FreeBody):
    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 0, drag_coef: float = 0) -> None: ...


class Ray(FreeBody):
    x: float
    y: float
    def __init__(self, id: int, plane: CartesianPlane, length: float, max_speed: float = 0, drag_coef: float = 0) -> None: ...
    def reset(self) -> None: ...


class Ball(FreePolygonBody):
    is_free: bool
    is_out: bool
    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 0, drag_coef: float = 0) -> None: ...
    def reset(self, pos: tuple) -> None: ...


class Player(DynamicPolygonBody):
    PLAYER_SIZE: int
    PLAYER_MAX_SPEED: int
    PLAYER_SPEED_BALL: int
    PLAYER_MAX_TURN_RATE: int
    PLAYER_MAX_FOV: int
    team_id: int
    kicked: bool
    has_ball: bool
    def __init__(self, id: int, team_id: int, plane: CartesianPlane, ability_point: float = 0.95) -> None: ...
    def reset(self, position: tuple, diraction: float) -> None: ...
    def kick(self, ball: FreePolygonBody, power: float) -> None: ...


class GoalKeeper(DynamicPolygonBody):
    PLAYER_SIZE: int
    PLAYER_MAX_SPEED: int
    PLAYER_SPEED_BALL: int
    PLAYER_MAX_TURN_RATE: int
    PLAYER_MAX_FOV: int
    team_id: int
    kicked: bool
    has_ball: bool
    def __init__(self, id: int, team_id: int, plane: CartesianPlane, ability_point: float = 1.0) -> None: ...
    def reset(self, position: tuple, diraction: float) -> None: ...
    def kick(self, ball: FreePolygonBody, power: float) -> None: ...