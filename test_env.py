from Game import Game, core
from single_agent_envs import RLFootball, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
import numpy as np


class EnvPath(RLFootball):

    def __init__(self, window, size, fps, team_size, full: bool = True) -> None:
        super().__init__(window, size, fps, team_size, full)
        self.player_path = []
        self.player_speed = []
        self.ball_path = []

    def get_state(self):
        self.ball_path.append(self.plane.to_XY(self.ball.position()))
        self.player_path.append(self.plane.to_XY(self.players[0].position()))
        self.player_speed.append(self.players[0].speed())
        return super().get_state()

    def reset(self, random_ball=False):
        self.player_path = []
        self.player_speed = []
        self.ball_path.append(self.plane.to_XY(self.ball.position()))
        state = super().reset(random_ball)
        return state


class TestEnv(Game):

    def __init__(self, env_count: int = 1, title: str = 'Single Agent train', random_ball: bool = False) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 30
        self.set_window()
        self.set_title(title)
        self.env_count = env_count
        self.random_ball = random_ball
        self.envs: list[EnvPath] = []
        self.team_size = 1
        self.setup()

    def setup(self):
        for _ in range(self.env_count):
            self.envs.append(EnvPath(self.window, self.size, 30, self.team_size, False))
            self.envs[-1].reset(self.random_ball)

    def reset(self):
        states = np.zeros((self.env_count, STATE_SPACE_SIZE))
        for i in range(self.env_count):
            states[i] = self.envs[i].reset(random_ball=self.random_ball)
        return states

    def step(self, actions: np.ndarray):
        next_states = np.zeros((self.env_count, STATE_SPACE_SIZE))
        rewards = np.zeros(self.env_count)
        dones = np.zeros(self.env_count)
        for i in range(self.env_count):
            next_states[i], rewards[i], dones[i] = self.envs[i].step([actions[i]])
        self.loop_once()
        for i in range(self.env_count):
            for j in range(self.envs[i].sensors.__len__()):
                self.envs[i].sensors[j].reset()
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
        # for pos in self.envs[0].ball_path:
        #     core.draw.circle(self.window, (255, 0, 0), pos, 5)
        # ball_path = [self.envs[0].ball_path[0], self.envs[0].plane.to_XY(self.envs[0].ball.position())]
        # core.draw.lines(self.window, (255, 0, 0), False, ball_path, width=3)
        core.draw.lines(self.window, (34, 177, 76), False, self.envs[0].player_path, width=3)
