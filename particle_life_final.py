import math
import colorsys

import numba
import numpy as np
from numba import config, cuda
import pygame

config.CUDA_LOW_OCCUPANCY_WARNINGS = False

TPB = (16, 16)
TPB_LIN = 32

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
PANEL_WIDTH = 400
SLIDER_LEFT = 10
SLIDER_TOP = 10
SLIDER_WIDTH = 10
SLIDER_DIST = 20
SCALE = 1
ZOOM = 1

SCREENSHOT_PATH = 'img/screenshot'

TRAILS = True
TRAIL_DECAY = 0.
EDGES = True
REALISTIC_MODE = False

FIELD_WIDTH = 1000 // SCALE
FIELD_HEIGHT = 1000 // SCALE
N_PARTICLES = (100,) * 20
N_DIFF_PARTICLES = len(N_PARTICLES)

MAX_SPEED = 10
MAX_INIT_VEL = 1
FRICTION = 0.5
GUST = .0

N_PARAMS = 3
MIN_R = 30
MAX_R = 100
MAX_F = 1000

COLORS = [tuple(c * 255 for c in colorsys.hsv_to_rgb(i / (2 * np.pi), 1, 1)) for i in range(N_DIFF_PARTICLES)]


class Color:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    RED = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])


def init_params():
    params = []

    for _ in range(N_DIFF_PARTICLES ** 2):
        params.append(np.random.uniform(0., MIN_R))
        params.append(np.random.uniform(MIN_R, MAX_R))
        params.append(np.random.uniform(-MAX_F, MAX_F))

    return params


@cuda.jit
def get_force(dist, params, t_self, t_other):
    offset = (int((t_other + N_DIFF_PARTICLES * t_self) * N_PARAMS)
              if t_self <= t_other and REALISTIC_MODE else
              int((t_self + N_DIFF_PARTICLES * t_other) * N_PARAMS))

    if params[offset] <= dist < params[offset + 1]:
        return params[offset + 2]
    elif dist < params[offset]:
        return - 10 / (dist ** 2)
    else:
        return 0.


@numba.njit(fastmath=True)
def rotate_tuple(x, angle):
    return (x[0] * np.cos(angle) - x[1] * np.sin(angle),
            x[0] * np.sin(angle) + x[1] * np.cos(angle))


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
def update_particles(pos, vel, types, field, params, gust, pause, cc):
    block = cuda.shared.array(shape=(TPB_LIN, 3), dtype=numba.float64)

    i = cuda.grid(1)

    tx = cuda.threadIdx.x
    bpg = cuda.gridDim.x

    force_x, force_y = gust

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

            if diff <= 0:
                continue

            force_x += get_force(diff, params, types[i], block[y, 2]) * diff_x / diff
            force_y += get_force(diff, params, types[i], block[y, 2]) * diff_y / diff

        cuda.syncthreads()

    if not pause:
        vel[i, 0] += force_x
        vel[i, 1] += force_y

        speed = math.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2)

        # friction
        vel[i, 0] -= vel[i, 0] * FRICTION
        vel[i, 1] -= vel[i, 1] * FRICTION

        if speed >= MAX_SPEED:
            vel[i, 0] *= MAX_SPEED / speed
            vel[i, 1] *= MAX_SPEED / speed

        if EDGES:
            p1_x = (pos[i, 0] + vel[i, 0] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
            p1_y = (pos[i, 1] + vel[i, 1] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

            if 0 <= p1_x < FIELD_WIDTH:
                pos[i, 0] += vel[i, 0]
            else:
                vel[i, 0] *= -1.

            if 0 <= p1_y < FIELD_HEIGHT:
                pos[i, 1] += vel[i, 1]
            else:
                vel[i, 1] *= -1.
        else:
            pos[i, 0] += 1.
            pos[i, 1] += vel[i, 1]

    p_x = (pos[i, 0] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
    p_y = (pos[i, 1] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

    if 0 <= p_x < FIELD_WIDTH and 0 <= p_y < FIELD_HEIGHT:
        rgb = cc[types[i]]
        field[int(p_x), int(p_y), 0] = rgb[0]
        field[int(p_x), int(p_y), 1] = rgb[1]
        field[int(p_x), int(p_y), 2] = rgb[2]


class Model:
    def __init__(self):
        self.__n_particles = sum(N_PARTICLES)

        self.params = [0.] * N_PARAMS * N_DIFF_PARTICLES ** 2

        particles_pos = [(np.random.uniform(0., FIELD_WIDTH), np.random.uniform(0., FIELD_HEIGHT))
                         for _ in range(self.__n_particles)]
        self.__particles_pos = cuda.to_device(particles_pos)

        particles_vel = [rotate_tuple((1, 0), np.random.uniform(0., 2 * np.pi))
                         for _ in range(self.__n_particles)]
        particles_vel = [(v[0] * np.random.uniform(0., MAX_INIT_VEL),
                          v[1] * np.random.uniform(0., MAX_INIT_VEL))
                         for v in particles_vel]
        self.__particles_vel = cuda.to_device(particles_vel)

        particles_type = []
        for n in range(N_DIFF_PARTICLES):
            particles_type.extend([n for _ in range(N_PARTICLES[n])])
        self.__particles_type = cuda.to_device(particles_type)

        field = np.zeros((FIELD_WIDTH, FIELD_HEIGHT, 3), dtype=int)
        self.__field = cuda.to_device(field)

        self.__gust = (GUST, 0.)

        self.__colors_cuda = cuda.to_device(COLORS)

    @property
    def field(self):
        return self.__field.copy_to_host()

    @property
    def particles(self):
        return self.__particles_pos

    def get_type(self, index):
        return self.__particles_type[index]

    def set_param(self, index, value):
        if index % 3 == 0:
            self.params[index] = MIN_R * value / 100
        elif index % 3 == 1:
            self.params[index] = MAX_R * value / 100
        else:
            self.params[index] = MAX_F * 2 * value / 100 - MAX_F

    def clear(self):
        self.__n_particles = 0
        self.__particles_pos = cuda.to_device([])
        self.__particles_vel = cuda.to_device([])
        self.__particles_type = cuda.to_device([])

    def add_particle(self, x, y, t):
        self.__n_particles += 1
        particles_pos = self.__particles_pos.copy_to_host().tolist()
        particles_pos.append([x - PANEL_WIDTH, y])
        self.__particles_pos = cuda.to_device(particles_pos)
        particles_vel = self.__particles_vel.copy_to_host().tolist()
        particles_vel.append([0., 0.])
        self.__particles_vel = cuda.to_device(particles_vel)
        particles_type = self.__particles_type.copy_to_host().tolist()
        particles_type.append(t)
        self.__particles_type = cuda.to_device(particles_type)

    def update(self, pause):
        bpg_x = math.ceil(FIELD_WIDTH / TPB[0])
        bpg_y = math.ceil(FIELD_HEIGHT / TPB[1])
        if TRAILS:
            decay_trails[(bpg_x, bpg_y), TPB](self.__field)
        else:
            clear_field[(bpg_x, bpg_y), TPB](self.__field)

        if self.__n_particles == 0:
            return

        self.__gust = rotate_tuple(self.__gust, np.random.normal(scale=0.05) * 2 * np.pi)

        bpg = math.ceil(self.__n_particles / TPB_LIN)
        params = cuda.to_device(self.params)
        update_particles[bpg, TPB_LIN](
            self.__particles_pos, self.__particles_vel,
            self.__particles_type, self.__field, params,
            self.__gust, pause, self.__colors_cuda
        )


class Slider:
    def __init__(self, index):
        self.index = index
        self.__x = SLIDER_LEFT
        self.y = SLIDER_TOP + index * SLIDER_DIST
        self.size = SLIDER_WIDTH
        self.__value = 0
        self.max_value = 100

    @property
    def x(self):
        return self.__x + self.__value

    @property
    def left(self):
        return self.__x

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if 0 <= value < self.max_value:
            self.__value = value
        elif value < self.__x:
            self.__value = 0
        else:
            self.__value = self.max_value

    def set_middle(self):
        self.__value = self.max_value / 2

    def clicked(self, x, y):
        if self.x <= x <= self.x + self.size and self.y <= y <= self.y + self.size:
            return True
        else:
            return False


class View:
    def __init__(self, params):
        self.__screen = pygame.display.set_mode((SCREEN_WIDTH + PANEL_WIDTH, SCREEN_HEIGHT))
        self.__clock = pygame.time.Clock()
        self.__scroll_level = 0
        self.__scroll_step = 50
        self.__sliders = [Slider(i) for i in range(params)]

    def set_sliders(self, params):
        for i in range(len(params)):
            if i % 3 == 0:
                self.__sliders[i].value = 100 * params[i] / MIN_R
            elif i % 3 == 1:
                self.__sliders[i].value = 100 * params[i] / MAX_R
            else:
                self.__sliders[i].value = 100 * (params[i] + MAX_F) / (2 * MAX_F)

    def get_slider_index(self, x, y):
        for slider in self.__sliders:
            if slider.clicked(x, y + self.__scroll_level):
                return slider.index

        return -1

    def update(self, model):
        self.__clock.tick(60)

        # draw slider panel

        self.__screen.fill(Color.BLACK)

        panel = pygame.Surface((PANEL_WIDTH, SCREEN_HEIGHT))

        y_start = 0
        for i in range(N_DIFF_PARTICLES):
            y = SLIDER_TOP + N_PARAMS * (i + 1) * SLIDER_DIST * N_DIFF_PARTICLES \
                - SLIDER_WIDTH // 2 - self.__scroll_level
            rect = pygame.Rect(PANEL_WIDTH // 2, y_start, PANEL_WIDTH, y - y_start)
            pygame.draw.rect(panel, COLORS[i], rect)
            y_start = y

        for slider in self.__sliders:
            y = slider.y + slider.size // 2 - self.__scroll_level
            hue = (slider.index // N_PARAMS) % N_DIFF_PARTICLES
            pygame.draw.line(panel, COLORS[hue], (0, y), (PANEL_WIDTH // 2, y))
            rect = pygame.Rect(slider.x, slider.y - self.__scroll_level, slider.size, slider.size)
            pygame.draw.rect(panel, Color.WHITE, rect)

        self.__screen.blit(panel, (0, 0))

        # draw particle field

        image = pygame.surfarray.make_surface(model.field)

        pygame.draw.line(image, Color.RED, (0, 0), (0, SCREEN_HEIGHT))

        zoomed = pygame.transform.scale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__screen.blit(zoomed, (PANEL_WIDTH, 0))

        pygame.display.flip()

    def scroll(self, d):
        if d <= 0:
            self.__scroll_level += self.__scroll_step
        elif self.__scroll_level > 0 and d > 0:
            self.__scroll_level -= self.__scroll_step

    def screenshot(self, file):
        pygame.image.save(self.__screen, file)


class Controller:
    def __init__(self):
        self.__model = Model()
        self.__view = View(len(self.__model.params))

        pygame.init()

    def run(self):
        running, pause = True, False
        screenshot_ctr = 0
        draw = False
        draw_type = 0

        clicked_slider = -1

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        pause = not pause
                    if event.key == pygame.K_s:
                        file = f'{SCREENSHOT_PATH}{screenshot_ctr}.png'
                        self.__view.screenshot(file)
                        screenshot_ctr += 1
                    elif event.key == pygame.K_1:
                        draw_type = 0
                    elif event.key == pygame.K_2:
                        draw_type = 1
                    elif event.key == pygame.K_3:
                        draw_type = 2
                    elif event.key == pygame.K_4:
                        draw_type = 3
                    elif event.key == pygame.K_SPACE:
                        self.__model.params = init_params()
                        self.__view.set_sliders(self.__model.params)
                    elif event.key == pygame.K_ESCAPE:
                        self.__model.clear()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_LEFT:
                        draw = True
                        clicked_slider = self.__view.get_slider_index(*pygame.mouse.get_pos())
                    elif event.button == pygame.BUTTON_RIGHT:
                        index = self.__view.get_slider_index(*pygame.mouse.get_pos())
                        self.__model.set_param(index, 50)
                        self.__view.set_sliders(self.__model.params)

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == pygame.BUTTON_LEFT:
                        draw = False
                        clicked_slider = -1

                if event.type == pygame.MOUSEWHEEL:
                    self.__view.scroll(event.y)

            if clicked_slider != -1:
                self.__model.set_param(
                    clicked_slider,
                    pygame.mouse.get_pos()[0] - SLIDER_LEFT
                )
                self.__view.set_sliders(self.__model.params)
            elif draw:
                x, y = pygame.mouse.get_rel()

                if abs(x) > 1 or abs(y) > 1:
                    self.__model.add_particle(*pygame.mouse.get_pos(), draw_type)

            self.__model.update(pause)

            self.__view.update(self.__model)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    Controller().run()
