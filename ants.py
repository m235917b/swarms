import random
import math
import os

import numba
import numpy as np
import pygame
from scipy.ndimage import gaussian_filter
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 1

TPB = (16, 16)

RANDOM_START = False

N_AGENTS = (100000, 0, 0)
VEL_AGENTS = 1
TRAIL_DECAY = 0.99
TRAIL_LIMIT = 0.0
FIELD_WIDTH = SCREEN_WIDTH // SCALE
FIELD_HEIGHT = SCREEN_WIDTH // SCALE
SENSOR_DIST = 10
SENSOR_RAD = 3
TURN_AMOUNT = 0.5 * np.pi * 0.5
RND_CHANCE = 0
RND_ANGLE_RANGE = 0.5 * np.pi
CHANCE_LEFT_OR_RIGHT = 0.7
CHANCE_LEFT_RIGHT = 0.3


class Color:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    RED = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])


@numba.njit(fastmath=True)
def rotate_tuple(x, angle):
    return x[0] * np.cos(angle) - x[1] * np.sin(angle), x[0] * np.sin(angle) + x[1] * np.cos(angle)


@cuda.jit
def decay_trails(trails):
    i, j = cuda.grid(2)

    if i < trails.shape[0] and j < trails.shape[1]:
        trails[i, j, 0] *= TRAIL_DECAY
        trails[i, j, 1] *= TRAIL_DECAY
        trails[i, j, 2] *= TRAIL_DECAY


@cuda.jit
def avg(arr):
    if arr.shape != (SENSOR_RAD * 2, SENSOR_RAD * 2):
        return 0.

    s, ctr = 0., 0

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not math.isnan(arr[i, j]) and arr[i, j] >= TRAIL_LIMIT:
                s += arr[i, j]
                ctr += 1

    return s / ctr if ctr > 0 else 0.


@cuda.jit
def sum_arr(arr):
    s = 0.

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not math.isnan(arr[i, j]) and arr[i, j] >= TRAIL_LIMIT:
                s += arr[i, j]

    return s


@cuda.jit
def rotate_cuda(x, angle):
    x[0] = x[0] * math.cos(angle) - x[1] * math.sin(angle)
    x[1] = x[0] * math.sin(angle) + x[1] * math.cos(angle)
    length = math.sqrt(x[0] ** 2 + x[1] ** 2)
    x[0] *= VEL_AGENTS / length
    x[1] *= VEL_AGENTS / length


@cuda.jit
def update_agents(pos, vel, types, field, rng_states):
    i = cuda.grid(1)

    if i < pos.shape[0]:
        front = (int(pos[i, 0] + vel[i, 0] * SENSOR_DIST),
                 int(pos[i, 1] + vel[i, 1] * SENSOR_DIST))
        left = (int(pos[i, 0] - vel[i, 1] * SENSOR_DIST),
                int(pos[i, 1] + vel[i, 0] * SENSOR_DIST))
        right = (int(pos[i, 0] + vel[i, 1] * SENSOR_DIST),
                 int(pos[i, 1] - vel[i, 0] * SENSOR_DIST))

        avg_front_0 = avg(field[(front[0] - SENSOR_RAD):(front[0] + SENSOR_RAD),
                          (front[1] - SENSOR_RAD):(front[1] + SENSOR_RAD), 0])
        avg_front_1 = avg(field[(front[0] - SENSOR_RAD):(front[0] + SENSOR_RAD),
                          (front[1] - SENSOR_RAD):(front[1] + SENSOR_RAD), 1])
        avg_front_2 = avg(field[(front[0] - SENSOR_RAD):(front[0] + SENSOR_RAD),
                          (front[1] - SENSOR_RAD):(front[1] + SENSOR_RAD), 2])

        avg_left_0 = avg(field[(left[0] - SENSOR_RAD):(left[0] + SENSOR_RAD),
                         (left[1] - SENSOR_RAD):(left[1] + SENSOR_RAD), 0])
        avg_left_1 = avg(field[(left[0] - SENSOR_RAD):(left[0] + SENSOR_RAD),
                         (left[1] - SENSOR_RAD):(left[1] + SENSOR_RAD), 1])
        avg_left_2 = avg(field[(left[0] - SENSOR_RAD):(left[0] + SENSOR_RAD),
                         (left[1] - SENSOR_RAD):(left[1] + SENSOR_RAD), 2])

        avg_right_0 = avg(field[(right[0] - SENSOR_RAD):(right[0] + SENSOR_RAD),
                          (right[1] - SENSOR_RAD):(right[1] + SENSOR_RAD), 0])
        avg_right_1 = avg(field[(right[0] - SENSOR_RAD):(right[0] + SENSOR_RAD),
                          (right[1] - SENSOR_RAD):(right[1] + SENSOR_RAD), 1])
        avg_right_2 = avg(field[(right[0] - SENSOR_RAD):(right[0] + SENSOR_RAD),
                          (right[1] - SENSOR_RAD):(right[1] + SENSOR_RAD), 2])

        averages = ((avg_front_0, avg_left_0, avg_right_0),
                    (avg_front_1, avg_left_1, avg_right_1),
                    (avg_front_2, avg_left_2, avg_right_2))

        angle = 0.
        rnd = xoroshiro128p_uniform_float32(rng_states, i)
        rnd_turn = xoroshiro128p_uniform_float32(rng_states, i)
        rnd_angle = xoroshiro128p_uniform_float32(rng_states, i)
        if rnd_turn < RND_CHANCE:
            angle += RND_ANGLE_RANGE * (2 * rnd_angle - 1)
        else:
            for x in range(len(averages)):
                if x == types[i]:
                    # angle += TURN_AMOUNT * (1 / (256 - averages[x][0]) - 1 / (256 - averages[x][1]))
                    if (averages[x][0] < averages[x][1] and averages[x][0] < averages[x][2]
                            and rnd < CHANCE_LEFT_OR_RIGHT):
                        if rnd_angle <= 0.5:
                            angle += TURN_AMOUNT
                        else:
                            angle -= TURN_AMOUNT
                    elif ((averages[x][0] < averages[x][1] or averages[x][0] < averages[x][2])
                          and rnd < CHANCE_LEFT_RIGHT):
                        if averages[x][1] > averages[x][2]:
                            angle += TURN_AMOUNT
                        elif averages[x][1] < averages[x][2]:
                            angle -= TURN_AMOUNT
                else:
                    if averages[x][0] < averages[x][1] and averages[x][0] < averages[x][2]:
                        if rnd_angle < 0.5:
                            angle -= TURN_AMOUNT
                        else:
                            angle += TURN_AMOUNT
                    elif averages[x][0] < averages[x][1] or averages[x][0] < averages[x][2]:
                        if averages[x][1] > averages[x][2]:
                            angle -= TURN_AMOUNT
                        elif averages[x][1] < averages[x][2]:
                            angle += TURN_AMOUNT
        rotate_cuda(vel[i], angle)

        if 0 <= pos[i, 0] + vel[i, 0] < FIELD_WIDTH:
            pos[i, 0] += vel[i, 0]
        else:
            vel[i, 0] *= -1.

        if 0 <= pos[i, 1] + vel[i, 1] < FIELD_HEIGHT:
            pos[i, 1] += vel[i, 1]
        else:
            vel[i, 1] *= -1.

        field[int(pos[i, 0]), int(pos[i, 1]), types[i]] = 255


class Model:
    def __init__(self):
        if RANDOM_START:
            agents_pos = [(np.random.uniform(0, 1) * FIELD_WIDTH, np.random.uniform(0, 1) * FIELD_HEIGHT)
                          for _ in range(N_AGENTS[0] + N_AGENTS[1] + N_AGENTS[2])]
        else:
            agents_pos = [(FIELD_WIDTH / 2, FIELD_HEIGHT / 2)
                          for _ in range(N_AGENTS[0] + N_AGENTS[1] + N_AGENTS[2])]
        self.__agents_pos = cuda.to_device(agents_pos)

        agents_vel = [rotate_tuple((VEL_AGENTS, 0), random.random() * 2 * np.pi)
                      for _ in range(N_AGENTS[0] + N_AGENTS[1] + N_AGENTS[2])]
        self.__agents_vel = cuda.to_device(agents_vel)

        agents_type = [0 for _ in range(N_AGENTS[0])]
        agents_type.extend([1 for _ in range(N_AGENTS[1])])
        agents_type.extend([2 for _ in range(N_AGENTS[2])])
        # for i in range(len(agents_vel)):
        # agents_type[i] = 0 if agents_vel[i][0] >= 0 else 1
        self.__agents_type = cuda.to_device(agents_type)

        field = np.zeros((FIELD_WIDTH, FIELD_HEIGHT, 3), dtype=float)
        self.__field = cuda.to_device(field)

        seed = int.from_bytes(os.urandom(8), byteorder='big')
        self.__rng_states = create_xoroshiro128p_states(self.__agents_pos.size, seed=seed)

    @property
    def trails(self):
        return self.__field.copy_to_host()

    def update(self):
        bpg_x = math.ceil(self.__field.shape[0] / TPB[0])
        bpg_y = math.ceil(self.__field.shape[1] / TPB[1])
        decay_trails[(bpg_x, bpg_y), TPB](self.__field)

        bpg = math.ceil(self.__agents_pos.size / TPB[0])
        update_agents[bpg, TPB[0]](self.__agents_pos, self.__agents_vel,
                                   self.__agents_type, self.__field, self.__rng_states)


def smooth(screen):
    """Apply a gaussian filter to each colour plane"""
    # Get reference pixels for each colour plane and then apply filter
    r = pygame.surfarray.pixels_red(screen)
    gaussian_filter(r, sigma=1, mode="nearest", output=r)
    g = pygame.surfarray.pixels_green(screen)
    gaussian_filter(g, sigma=1, mode="nearest", output=g)
    b = pygame.surfarray.pixels_blue(screen)
    gaussian_filter(b, sigma=1, mode="nearest", output=b)


class View:
    def __init__(self):
        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__clock = pygame.time.Clock()

    def update(self, model):
        # self.__clock.tick(60)

        self.__screen.fill(Color.BLACK)

        image = pygame.surfarray.make_surface(model.trails)

        zoomed = pygame.transform.scale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__screen.blit(zoomed, (0, 0))
        smooth(self.__screen)

        pygame.display.flip()

        # print(self.__clock.get_fps())

    def screenshot(self, file):
        pygame.image.save(self.__screen, file)


class Controller:
    def __init__(self):
        self.__view = View()
        self.__model = Model()

        pygame.init()

    def run(self):
        running = True
        screenshot_ctr = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        file = f'img/dc{TRAIL_DECAY}-t{TURN_AMOUNT}-rnd{RND_CHANCE}' \
                               f'-rnda{RND_ANGLE_RANGE}-{screenshot_ctr}'
                        file = file.replace('.', '_') + '.png'
                        self.__view.screenshot(file)
                        screenshot_ctr += 1

            self.__model.update()

            self.__view.update(self.__model)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    Controller().run()
