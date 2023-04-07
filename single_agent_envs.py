from Game import Game
from Game import core
from Game.graphic import CartesianPlane
from Game.physics import StaticRectangleBody
from football import Football, NOOP, BALL_SPEED_MAX, STOP, GO_FORWARD, TURN_LEFT, TURN_RIGHT, KICK, GOAL_AREA_WIDTH, RAY_COUNT
import numpy as np


ACTION_SPACE_SIZE = 6
STATE_SPACE_SIZE = 9 + RAY_COUNT


class RLFootball(Football):

    def __init__(self, window, size, fps, team_size, full: bool = True) -> None:
        super().__init__(window, size, fps, team_size, full)
        self.counter = 0
        self.done = False

    def step(self, actions: list = None):
        super().step(actions)
        self.counter += 1
        if self.counter == (self.fps * 10):
            self.done = True
        ball_pos = self.plane.to_xy(self.ball.position())
        if self.ball.is_out or ball_pos[0] < 0 or ball_pos[0] > self.size[0] // 2 - GOAL_AREA_WIDTH \
                or ball_pos[1] < -self.size[1] // 2 + GOAL_AREA_WIDTH // 2 or ball_pos[1] > self.size[1] // 2 - GOAL_AREA_WIDTH // 2:
            self.done = True
        if self.teamRight.score:
            self.done = True
            reward = 300
        elif self.done and not self.teamRight.score:
            reward = -1
        else:
            reward = -1
        return self.get_state(), reward, self.done

    def get_state(self):
        state = []
        state.extend(self.sensors[0].get_state())
        ball_pos = self.plane.to_xy(self.ball.position())
        ball_dir = self.ball.direction() / 360
        ball_spd = self.ball.speed() / BALL_SPEED_MAX
        player_pos = self.plane.to_xy(self.players[0].position())
        player_dir = self.players[0].direction() / 360
        player_spd = self.players[0].speed() / self.players[0].PLAYER_MAX_SPEED
        has_ball = self.players[0].has_ball
        state.extend([ball_pos[0] / self.plane.x_max,
                      ball_pos[1] / self.plane.y_max,
                      ball_dir,
                      ball_spd,
                      player_pos[0] / self.plane.x_max,
                      player_pos[1] / self.plane.y_max,
                      player_dir,
                      player_spd,
                      has_ball])
        return np.array(state)

    def reset(self, random_ball=False):
        self.counter = 0
        self.done = False
        x = 300
        y = 0
        if random_ball:
            y_lim = (self.plane.window_size[1] - GOAL_AREA_WIDTH) / 2
            x = np.random.randint(0, self.plane.x_max - GOAL_AREA_WIDTH + 1)
            y = np.random.randint(-y_lim, y_lim + 1)
        self.ball.reset((x, y))
        self.teamRight.reset()
        self.engine.step()
        self.ball.step()
        self.check_ball()
        return self.get_state()

    def create_wall(self, wall_width=120, wall_height=5):
        y = self.size[1] // 2 - wall_width // 2 - wall_height // 2
        pad = 40
        for _ in range(self.size[1] // wall_width):
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width),
                                                   self.plane.createVector(-pad // 2, y)),
                                    (wall_height, wall_width)))
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width),
                                                   self.plane.createVector(self.size[0] // 2 - GOAL_AREA_WIDTH + pad, y)),
                                    (wall_height, wall_width)))
            y -= wall_width

        x = 0
        for _ in range(self.size[0] // wall_width):
            vec = self.plane.createVector(x, self.size[1] // 2 - GOAL_AREA_WIDTH // 2 + pad)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width), vec),
                                    (wall_width, wall_height)))
            vec = self.plane.createVector(x, -self.size[1] // 2 + GOAL_AREA_WIDTH // 2 - pad)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width), vec),
                                    (wall_width, wall_height)))
            x += wall_width


class SinglePlayerFootball(Game):

    def __init__(self, title: str = 'Single Agent train') -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 120
        self.set_window()
        self.set_title(title)
        self.football: RLFootball = None
        self.team_size = 3
        self.setup()

    def setup(self):
        self.football = RLFootball(self.window, self.size, self.fps, self.team_size, False)

    def reset(self, random_ball=False):
        return self.football.reset(random_ball=random_ball)

    def step(self, action: int = NOOP):
        return self.football.step([action])

    def loop(self):
        actions = [NOOP for _ in range(self.team_size)]  # +2 goal keeper agents
        idx = self.football.current_player
        if self.keys[core.K_UP]:
            actions[idx] = GO_FORWARD
        if self.keys[core.K_DOWN]:
            actions[idx] = STOP
        if self.keys[core.K_LEFT]:
            actions[idx] = TURN_LEFT
        if self.keys[core.K_RIGHT]:
            actions[idx] = TURN_RIGHT
        if self.keys[core.K_f]:
            actions[idx] = KICK
        s, r, d = self.football.step(actions)
        if d:
            self.reset()

    def loop_once(self):
        super().loop_once()
        return self.football.done

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
                self.football.done = True
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering

    def onRender(self):
        self.window.fill((255, 255, 255))
        self.football.show()


class SinglePlayerFootballParallel(Game):

    def __init__(self, env_count: int = 1, title: str = 'Single Agent train') -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 30
        self.set_window()
        self.set_title(title)
        self.env_count = env_count
        self.envs: list[RLFootball] = []
        self.team_size = 1
        self.setup()

    def setup(self):
        for _ in range(self.env_count):
            self.envs.append(RLFootball(self.window, self.size, self.fps, self.team_size, False))

    def reset(self, random_ball=False):
        states = np.zeros((self.env_count, STATE_SPACE_SIZE))
        for i in range(self.env_count):
            states[i] = self.envs[i].reset(random_ball=random_ball)
        return states

    def step(self, actions: np.ndarray):
        next_states = np.zeros((self.env_count, STATE_SPACE_SIZE))
        rewards = np.zeros(self.env_count)
        dones = np.zeros(self.env_count)
        for i in range(self.env_count):
            next_states[i], rewards[i], dones[i] = self.envs[i].step([actions[i]])
        self.loop_once()
        return next_states, rewards, dones

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
                for env in self.envs:
                    env.done = True
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering

    def onRender(self):
        self.window.fill((255, 255, 255))
        for i in range(self.env_count):
            self.envs[i].show()
