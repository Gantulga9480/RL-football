from typing import Union

_vector2d_or_tuple = Union[vector2d, tuple]


class scalar:
    min: float
    max: float
    def __init__(self, value: float, limits: tuple) -> None: ...
    @property
    def value(self) -> float: ...
    @value.setter
    def value(self, o: float) -> None: ...


class point2d:
    def __init__(self, x: float, y: float, x_lim: tuple = None, y_lim: tuple = None) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, o: float) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, o: float) -> None: ...
    @property
    def xy(self) -> tuple: ...
    @xy.setter
    def xy(self, o: object) -> None: ...
    def set_x_ref(self, o: scalar) -> None: ...
    def set_y_ref(self, o: scalar) -> None: ...
    def get_x_ref(self) -> scalar: ...
    def get_y_ref(self) -> scalar: ...


class vector2d:
    min: float
    max: float
    def __init__(self, x: float, y: float, max_length: float = 0, min_length: float = 0) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, o: float) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, o: float) -> None: ...
    @property
    def head(self) -> tuple: ...
    @head.setter
    def head(self, o: tuple) -> None: ...
    def add(self, o: float) -> None: ...
    def scale(self, o: float) -> None: ...
    def set_x_ref(self, o: scalar) -> None: ...
    def set_y_ref(self, o: scalar) -> None: ...
    def set_head_ref(self, o: point2d) -> None: ...
    def get_x_ref(self) -> scalar: ...
    def get_y_ref(self) -> scalar: ...
    def get_head_ref(self) -> point2d: ...
    def rotate(self, radians: float) -> None: ...
    def mag(self) -> float: ...
    def dir(self) -> float: ...
    def distance_to(self, vector: vector2d) -> float: ...
    def angle_between(self, vector: vector2d) -> float: ...
    def dot(self, vector: vector2d) -> float: ...
    def unit(self, scale: float = 1, vector: bool = True) -> _vector2d_or_tuple: ...
    def normal(self, scale: float = 1, vector: bool = True) -> _vector2d_or_tuple: ...
