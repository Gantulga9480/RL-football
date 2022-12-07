from Game.graphic.cartesian import CartesianPlane, Vector2d
from Game.color import BLACK
from pygame.color import Color
from pygame import Surface
import numpy as np
from typing import Union


class Shape:
    window: Surface
    plane: CartesianPlane
    vertex_count: int
    color: tuple
    vertices: list[Vector2d]
    def __init__(self, plane: CartesianPlane) -> None: ...
    def rotate(self, angle) -> None: ...
    def scale(self, factor) -> None: ...
    def show(self, show_vertex: bool = False) -> None: ...


class Line(Shape):
    def __init__(self, plane: CartesianPlane, length: float) -> None: ...


class Rectangle(Shape):
    def __init__(self, plane: CartesianPlane, shape: tuple) -> None: ...


class Triangle(Shape):
    def __init__(self, plane: CartesianPlane, shape: tuple) -> None: ...


class Polygon(Shape):
    def __init__(self, plane: CartesianPlane, shape: tuple) -> None: ...
