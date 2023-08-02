from Game import Game, core
from Game.graphic import CartesianPlane, Polygon


class Test(Game):

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)

    def setup(self):
        self.plane = CartesianPlane(self.window, self.size)
        self.shape_plane = self.plane.createPlane(100, 100)
        self.shape = Polygon(self.shape_plane, (50,) * 5)
        self.shape.color = (0, 0, 0)

        self.second_shape_plane = self.shape_plane.createPlane(-100, 100)
        self.second_shape = Polygon(self.second_shape_plane, (30,) * 5)

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False

    def onRender(self):
        self.window.fill((255, 255, 255))
        self.shape.show(show_vertex=True)
        self.second_shape.show(show_vertex=True)
        self.shape_plane.get_parent_vector().show()
        self.second_shape_plane.get_parent_vector().show()
        self.plane.show()
        self.shape_plane.show()
        self.second_shape_plane.show()


Test().loop_forever()
