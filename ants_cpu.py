from collections import namedtuple
import random

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter
from numba import njit, cuda, from_dtype

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 1

N_AGENTS = 100000
TRAIL_DECAY = 0.99
TRAIL_VANISH = 100
FIELD_WIDTH = SCREEN_WIDTH // SCALE
FIELD_HEIGHT = SCREEN_WIDTH // SCALE
SENSOR_DIST = 5
SENSOR_RAD = 3
TURN_AMOUNT = 50


class Color:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    RED = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])


def rotate(x, angle):
    return np.array([x[0] * np.cos(angle) - x[1] * np.sin(angle),
                     x[0] * np.sin(angle) + x[1] * np.cos(angle)])


class Agent:
    def __init__(self, x, y, angle, colony):
        self.__pos = np.array([x, y])
        self.__vel = rotate(np.array([1, 0]), angle)
        self.colony = colony

    @property
    def pos(self):
        return self.__pos

    @property
    def left(self):
        return (self.__pos + np.array([-self.__vel[1], self.__vel[0]]) * SENSOR_DIST).astype(int)

    @property
    def right(self):
        return (self.__pos + np.array([self.__vel[1], -self.__vel[0]]) * SENSOR_DIST).astype(int)

    def update(self, left, right):
        angle = 0

        for i in range(left.shape[0]):
            if i == self.colony:
                angle += TURN_AMOUNT * (1 / (256 - left[i]) - 1 / (256 - right[i]))
            else:
                angle -= TURN_AMOUNT * (1 / (256 - left[i]) - 1 / (256 - right[i]))

        self.__vel = rotate(self.__vel, angle)
        self.__pos += self.__vel

        if self.__pos[0] < 0 or self.__pos[0] >= FIELD_WIDTH:
            self.__pos -= self.__vel
            self.__vel[0] *= -1

        if self.__pos[1] < 0 or self.__pos[1] >= FIELD_HEIGHT:
            self.__pos -= self.__vel
            self.__vel[1] *= -1


class Model:
    def __init__(self):
        self.__agents = [Agent(float(random.randint(0, FIELD_WIDTH - 1)),
                               float(random.randint(0, FIELD_HEIGHT - 1)),
                               random.random() * 2 * np.pi, 0) for _ in range(N_AGENTS)]
        """self.__agents.extend([Agent(float(random.randint(0, FIELD_WIDTH - 1)),
                                    float(random.randint(0, FIELD_HEIGHT - 1)),
                                    random.random() * 2 * np.pi, 1) for _ in range(N_AGENTS)])"""
        """self.__agents = [Agent(100., 100., random.random() * 2 * np.pi, 0) for _ in range(N_AGENTS)]
        self.__agents.extend([Agent(400., 400., random.random() * 2 * np.pi, 1) for _ in range(N_AGENTS)])"""
        """self.__agents = [Agent(100., float(random.randint(0, FIELD_HEIGHT - 1)),
                               random.random() * 2 * np.pi, 0) for _ in range(N_AGENTS)]
        self.__agents.extend([Agent(400., float(random.randint(0, FIELD_HEIGHT - 1)),
                                    random.random() * 2 * np.pi, 1) for _ in range(N_AGENTS)])"""
        """self.__agents = []
        for _ in range(N_AGENTS):
            pos_0 = rotate(np.array([float(random.randint(0, 50)), 0]), random.random() * 2 * np.pi)
            pos_0 += np.array([FIELD_WIDTH / 2, FIELD_HEIGHT / 2])
            pos_1 = rotate(np.array([float(random.randint(50, 100)), 0]), random.random() * 2 * np.pi)
            pos_1 += np.array([FIELD_WIDTH / 2, FIELD_HEIGHT / 2])
            self.__agents.append(Agent(pos_0[0], pos_0[1], random.random() * 2 * np.pi, 0))
            self.__agents.append(Agent(pos_1[0], pos_1[1], random.random() * 2 * np.pi, 1))"""

        self.__trails = np.zeros((FIELD_WIDTH, FIELD_HEIGHT, 3), dtype=float)

    @property
    def agents(self):
        return self.__agents

    @property
    def trails(self):
        return self.__trails

    def update(self):
        self.__trails *= TRAIL_DECAY

        for agent in self.__agents:
            self.__trails[int(agent.pos[0]), int(agent.pos[1]), agent.colony] = 255
            left = agent.left
            right = agent.right
            left_arr = self.__trails[abs(left[0] - SENSOR_RAD):abs(left[0] + SENSOR_RAD),
                       abs(left[1] - SENSOR_RAD):abs(left[1] + SENSOR_RAD)]
            right_arr = self.__trails[abs(right[0] - SENSOR_RAD):abs(right[0] + SENSOR_RAD),
                        abs(right[1] - SENSOR_RAD):abs(right[1] + SENSOR_RAD)]
            left_means = (np.array([0., 0., 0.])
                          if left_arr.shape[0] == 0 or left_arr.shape[1] == 0 else
                          np.array([left_arr[:, :, 0].mean(),
                                    left_arr[:, :, 1].mean(),
                                    left_arr[:, :, 2].mean()]))
            right_means = (np.array([0., 0., 0.])
                           if right_arr.shape[0] == 0 or right_arr.shape[1] == 0 else
                           np.array([right_arr[:, :, 0].mean(),
                                     right_arr[:, :, 1].mean(),
                                     right_arr[:, :, 2].mean()]))
            agent.update(left_means, right_means)
            """agent.update(0 if left_arr.shape[0] == 0 or left_arr.shape[1] == 0 else left_arr.mean(),
                         0 if right_arr.shape[0] == 0 or right_arr.shape[1] == 0 else right_arr.mean())"""


def smooth(screen):
    """Apply a gaussian filter to each colour plane"""
    # Get reference pixels for each colour plane and then apply filter
    r = pygame.surfarray.pixels_red(screen)
    gaussian_filter(r, sigma=1, mode="nearest", output=r)
    g = pygame.surfarray.pixels_green(screen)
    gaussian_filter(g, sigma=0, mode="nearest", output=g)
    b = pygame.surfarray.pixels_blue(screen)
    gaussian_filter(b, sigma=0, mode="nearest", output=b)


class View:
    def __init__(self):
        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__clock = pygame.time.Clock()

    def update(self, model):
        self.__clock.tick(30)

        self.__screen.fill(Color.BLACK)

        image = pygame.surfarray.make_surface(model.trails)

        for agent in model.agents:
            pygame.draw.circle(image, Color.WHITE, agent.pos, 1)

        zoomed = pygame.transform.scale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__screen.blit(zoomed, (0, 0))
        smooth(self.__screen)

        pygame.display.flip()

        print(self.__clock.get_fps())


class Controller:
    def __init__(self):
        self.__view = View()
        self.__model = Model()

        pygame.init()

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.__model.update()

            self.__view.update(self.__model)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    Controller().run()
