import pygame as pg    # noqa
from .color import *   # noqa


class Game:

    def __init__(self,
                 title: str = 'PyGameDemo',
                 width: int = 640,
                 height: int = 480,
                 fps: int = 60,
                 flags: int = 0,
                 render: bool = True) -> None:
        pg.init()

        # Main window
        self.width: int = width
        self.height: int = height
        self.fps: int = fps
        self.flags: int = flags
        self.title: str = title
        self.backgroundColor: pg.Color = WHITE
        self.running: bool = True
        self.rendering: bool = render
        self.clock = pg.time.Clock()

        # Event
        self.mouse_x = 0
        self.mouse_y = 0
        self.keys = []

        # Render
        self.sprites: list[pg.Rect] = []
        self.window = self.get_window(self.width, self.height, self.flags)

    def __del__(self):
        pg.quit()

    def mainloop(self):
        self.__highLevelSetup()
        while self.running:
            self.__highLevelEventHandler()
            self.USR_loop()
            self.__highLevelRender()

    def loop_once(self):
        self.__highLevelEventHandler()
        self.USR_loop()
        self.__highLevelRender()

    def USR_loop(self):
        """ User should override this method """
        ...

    def __highLevelSetup(self):
        self.set_title(self.title)
        self.USR_setup()

    def USR_setup(self):
        """ User should override this method """
        ...

    def __highLevelEventHandler(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                break
            else:
                self.USR_eventHandler(event)
        self.mouse_x, self.mouse_y = pg.mouse.get_pos()
        self.keys = pg.key.get_pressed()

    def USR_eventHandler(self, event):
        """ User should override this method """
        ...

    def __highLevelRender(self):
        if self.rendering and self.running:
            self.window.fill(self.backgroundColor)
            self.USR_render()
            pg.display.flip()
            self.clock.tick(self.fps)

    def USR_render(self):
        """ User should override this method """
        ...

    @staticmethod
    def set_title(title: str):
        pg.display.set_caption(title)

    @staticmethod
    def get_window(width: int, height: int, flags: int):
        """ Avoid calling outside of PyGameBase instance """
        return pg.display.set_mode((width, height), flags)
