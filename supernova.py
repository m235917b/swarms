import math

import numba
import numpy as np
from numba import njit, cuda
import pygame
from scipy.ndimage import gaussian_filter

TPB = (16, 16)
TPB_LIN = 64

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 1
ZOOM = 1e-7

TRAILS = False
TRAIL_DECAY = 0.999

FIELD_WIDTH = 1000
FIELD_HEIGHT = 1000
N_PARTICLES = (10000, 0, 0)
SPEEDUP = 10000
MAX_SPEED = 10000 * SPEEDUP
GRAVITY = 6.67e19 * SPEEDUP ** 2 / N_PARTICLES[0]


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
def clear_field(field):
    i, j = cuda.grid(2)

    if i < field.shape[0] and j < field.shape[1]:
        field[i, j, 0] = 0
        field[i, j, 1] = 0
        field[i, j, 2] = 0


@cuda.jit
def update_agents(pos, vel, types, field):
    block = cuda.shared.array(shape=(TPB_LIN, 3), dtype=numba.float64)

    i = cuda.grid(1)
    tx = cuda.threadIdx.x
    bpg = cuda.gridDim.x

    force_x, force_y = 0, 0

    for x in range(bpg):
        block[tx, 0] = pos[tx + x * TPB_LIN, 0]
        block[tx, 1] = pos[tx + x * TPB_LIN, 1]
        block[tx, 2] = types[tx + x * TPB_LIN]

        if i >= pos.shape[0]:
            return

        cuda.syncthreads()

        for y in range(TPB_LIN):
            if y + x * TPB_LIN >= pos.shape[0]:
                break

            diff_x = block[y, 0] - pos[i, 0]
            diff_y = block[y, 1] - pos[i, 1]
            diff = math.sqrt(diff_x ** 2 + diff_y ** 2)

            if diff < 1:
                continue

            if block[y, 2] != types[i]:
                force_x -= 1 / diff * 0.1 * diff_x / diff
                force_y -= 1 / diff * 0.1 * diff_y / diff
            else:
                force_x += (1 / diff ** 2) * GRAVITY * diff_x / diff
                force_y += (1 / diff ** 2) * GRAVITY * diff_y / diff

        cuda.syncthreads()

    vel[i, 0] += force_x
    vel[i, 1] += force_y

    """if vel[i, 0] > MAX_SPEED:
        vel[i, 0] = MAX_SPEED
    elif vel[i, 0] < -MAX_SPEED:
        vel[i, 0] = -MAX_SPEED

    if vel[i, 1] > MAX_SPEED:
        vel[i, 1] = MAX_SPEED
    elif vel[i, 1] < -MAX_SPEED:
        vel[i, 1] = -MAX_SPEED"""

    speed = math.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2)

    if speed >= MAX_SPEED:
        vel[i, 0] *= MAX_SPEED / speed
        vel[i, 1] *= MAX_SPEED / speed

    """p1_x = (pos[i, 0] + vel[i, 0] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
    p1_y = (pos[i, 1] + vel[i, 1] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

    if 0 <= p1_x < FIELD_WIDTH:
        pos[i, 0] += vel[i, 0]
    else:
        vel[i, 0] *= -1.

    if 0 <= p1_y < FIELD_HEIGHT:
        pos[i, 1] += vel[i, 1]
    else:
        vel[i, 1] *= -1."""

    pos[i, 0] += vel[i, 0]
    pos[i, 1] += vel[i, 1]

    p_x = (pos[i, 0] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
    p_y = (pos[i, 1] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

    if 0 <= p_x < FIELD_WIDTH and 0 <= p_y < FIELD_HEIGHT:
        field[int(p_x), int(p_y), types[i]] = 255

    # field[int(pos[i, 0] + 500 * vel[i, 0]), int(pos[i, 1] + 500 * vel[i, 1]), types[i]] = 255


class Model:
    def __init__(self):
        self.__n_particles = sum(N_PARTICLES)

        # particles_pos = [(FIELD_WIDTH / 2, FIELD_HEIGHT / 2) for _ in range(self.__n_particles)]
        particles_pos = [rotate_tuple((np.random.uniform() * 7e8, 0), np.random.uniform() * 2 * np.pi)
                         for _ in range(self.__n_particles)]
        particles_pos = [(i[0] + FIELD_WIDTH / 2, i[1] + FIELD_HEIGHT / 2) for i in particles_pos]
        self.__particles_pos = cuda.to_device(particles_pos)
        print(self.__particles_pos.shape)

        particles_vel = [(0., 0.)
                         for _ in range(self.__n_particles)]
        self.__particles_vel = cuda.to_device(particles_vel)

        particles_type = []
        for n in range(len(N_PARTICLES)):
            particles_type.extend([n for _ in range(N_PARTICLES[n])])
        self.__particles_type = cuda.to_device(particles_type)

        field = np.zeros((FIELD_WIDTH, FIELD_HEIGHT, 3), dtype=float)
        self.__field = cuda.to_device(field)

    @property
    def field(self):
        return self.__field.copy_to_host()

    def update(self):
        bpg_x = math.ceil(self.__field.shape[0] / TPB[0])
        bpg_y = math.ceil(self.__field.shape[1] / TPB[1])
        if TRAILS:
            decay_trails[(bpg_x, bpg_y), TPB](self.__field)
        else:
            clear_field[(bpg_x, bpg_y), TPB](self.__field)

        bpg = math.ceil(self.__n_particles / TPB_LIN)
        update_agents[bpg, TPB_LIN](self.__particles_pos, self.__particles_vel,
                                    self.__particles_type, self.__field)


def smooth(screen):
    """Apply a gaussian filter to each colour plane"""
    # Get reference pixels for each colour plane and then apply filter
    r = pygame.surfarray.pixels_red(screen)
    gaussian_filter(r, sigma=0, mode="nearest", output=r)
    g = pygame.surfarray.pixels_green(screen)
    gaussian_filter(g, sigma=0, mode="nearest", output=g)
    b = pygame.surfarray.pixels_blue(screen)
    gaussian_filter(b, sigma=0, mode="nearest", output=b)


class View:
    def __init__(self):
        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__clock = pygame.time.Clock()

    def update(self, model):
        self.__clock.tick(60)

        self.__screen.fill(Color.BLACK)

        image = pygame.surfarray.make_surface(model.field)

        zoomed = pygame.transform.scale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__screen.blit(zoomed, (0, 0))
        smooth(self.__screen)

        pygame.display.flip()

        print(self.__clock.get_fps())

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
                        file = f'img/trails{screenshot_ctr}.png'
                        self.__view.screenshot(file)
                        screenshot_ctr += 1

            self.__model.update()

            self.__view.update(self.__model)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    Controller().run()
