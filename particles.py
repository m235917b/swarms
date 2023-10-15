import math

import numba
import numpy as np
from numba import cuda
import pygame
from scipy.ndimage import gaussian_filter

TPB = (16, 16)
TPB_LIN = 32

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 1
ZOOM = 1

TRAILS = False
TRAIL_DECAY = 0.99

EDGES = True

FIELD_WIDTH = 1000 // SCALE
FIELD_HEIGHT = 1000 // SCALE
N_PARTICLES = (20, 20, 20)
MAX_SPEED = 10


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
def get_force(dist, params, t_self, t_other):
    offset = int((t_self + 2 * t_other) * 4)

    if dist < params[offset + 1]:
        return (dist - params[offset + 1]) * params[offset + 0] / params[offset + 1]
    elif dist < params[offset + 1] + (params[offset + 3] - params[offset + 1]) / 2:
        return ((dist - params[offset + 1]) * params[offset + 2]
                / (params[offset + 1] + (params[offset + 3] - params[offset + 1]) / 2 - params[offset + 1]))
    elif dist < params[offset + 3]:
        return ((dist - params[offset + 3]) * -params[offset + 2]
                / (params[offset + 3] - params[offset + 1] + (params[offset + 3] - params[offset + 1]) / 2))
    else:
        return -10 / (dist ** 2) if t_self == t_other else 0.

    """if dist < params[offset + 1]:
        return (dist - params[offset + 1]) * params[offset + 0] / params[offset + 1]
    elif dist < params[offset + 1] + (params[offset + 3] - params[offset + 1]) / 2:
        return ((dist - params[offset + 1]) * params[offset + 2]
                / (params[offset + 1] + (params[offset + 3] - params[offset + 1]) / 2 - params[offset + 1]))
    else:
        return ((dist - FIELD_WIDTH) * -params[offset + 2]
                / (FIELD_WIDTH - params[offset + 1] + (FIELD_WIDTH - params[offset + 1]) / 2))"""

    # return params[offset] * 3 / (dist ** 2) - 1 / (dist ** 3) - dist * params[offset + 1] * 0.0000001


"""@cuda.jit
def update_particles(pos, vel, types, field):
    block = cuda.shared.array(shape=(TPB_LIN, 3), dtype=numba.float64)

    i = cuda.grid(1)
    tx = cuda.threadIdx.x
    bpg = cuda.gridDim.x

    force_x, force_y = 0, 0
    p_type = types[i]
    p_pos_x = pos[i, 0]
    p_pos_y = pos[i, 1]
    p_vel_x = vel[i, 0]
    p_vel_y = vel[i, 1]

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

            diff_x = block[y, 0] - p_pos_x
            diff_y = block[y, 1] - p_pos_y
            diff = math.sqrt(diff_x ** 2 + diff_y ** 2)

            if diff <= 1:
                continue

            g = (1 / diff ** 2) * GRAVITY

            if block[y, 2] != p_type:
                force_x -= 1 / diff * 0.1 * diff_x / diff
                force_y -= 1 / diff * 0.1 * diff_y / diff
            else:
                force_x += g * diff_x / diff
                force_y += g * diff_y / diff

        cuda.syncthreads()

    p_vel_x += force_x
    p_vel_y += force_y

    speed = math.sqrt(p_vel_x ** 2 + p_vel_y ** 2)

    if speed >= MAX_SPEED:
        p_vel_x *= MAX_SPEED / speed
        p_vel_y *= MAX_SPEED / speed

    p1_x = (pos[i, 0] + vel[i, 0] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
    p1_y = (pos[i, 1] + vel[i, 1] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

    if 0 <= p1_x < FIELD_WIDTH:
        p_pos_x += p_vel_x
    else:
        p_vel_x *= -1.

    if 0 <= p1_y < FIELD_HEIGHT:
        p_pos_y += p_vel_y
    else:
        p_vel_y *= -1.

    p_pos_x += p_vel_x
    p_pos_y += p_vel_y

    p_x = (p_pos_x - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
    p_y = (p_pos_y - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

    if 0 <= p_x < FIELD_WIDTH and 0 <= p_y < FIELD_HEIGHT:
        field[int(p_x), int(p_y), p_type] = 255

    pos[i, 0] = p_pos_x
    pos[i, 1] = p_pos_y
    vel[i, 0] = p_vel_x
    vel[i, 1] = p_vel_y"""


@cuda.jit
def update_particles(pos, vel, types, field, params, gust):
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

    vel[i, 0] += force_x
    vel[i, 1] += force_y

    speed = math.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2)

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
        pos[i, 0] += vel[i, 0]
        pos[i, 1] += vel[i, 1]

    p_x = (pos[i, 0] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2
    p_y = (pos[i, 1] - FIELD_WIDTH / 2) * ZOOM + FIELD_WIDTH / 2

    if 0 <= p_x < FIELD_WIDTH and 0 <= p_y < FIELD_HEIGHT:
        field[int(p_x), int(p_y), types[i]] = 255

    # field[int(pos[i, 0] + 500 * vel[i, 0]), int(pos[i, 1] + 500 * vel[i, 1]), types[i]] = 255


@cuda.jit
def update_particles_ns(pos, vel, types, field):
    i = cuda.grid(1)

    if i >= pos.shape[0]:
        return

    force_x, force_y = 0, 0

    for x in range(pos.shape[0]):
        diff_x = pos[x, 0] - pos[i, 0]
        diff_y = pos[x, 1] - pos[i, 1]
        diff = math.sqrt(diff_x ** 2 + diff_y ** 2)

        if diff <= 1:
            continue

        if types[x] != types[i]:
            force_x -= 1 / diff * 0.1 * diff_x / diff
            force_y -= 1 / diff * 0.1 * diff_y / diff
        else:
            force_x += diff_x / diff
            force_y += diff_y / diff

    vel[i, 0] += force_x
    vel[i, 1] += force_y

    speed = math.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2)

    if speed >= MAX_SPEED:
        vel[i, 0] *= MAX_SPEED / speed
        vel[i, 1] *= MAX_SPEED / speed

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
        self.__params = [100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0,
                         100, 10, 0, 0]

        # particles_pos = [(FIELD_WIDTH / 2, FIELD_HEIGHT / 2) for _ in range(self.__n_particles)]
        """particles_pos = [rotate_tuple((np.random.uniform() * 100, 0), np.random.uniform() * 2 * np.pi)
                         for _ in range(self.__n_particles)]
        particles_pos = [(i[0] + FIELD_WIDTH / 2, i[1] + FIELD_HEIGHT / 2) for i in particles_pos]"""
        particles_pos = [(np.random.uniform() * FIELD_WIDTH, np.random.uniform() * FIELD_HEIGHT)
                         for _ in range(self.__n_particles)]
        self.__particles_pos = cuda.to_device(particles_pos)
        print(self.__particles_pos.shape)

        particles_vel = [rotate_tuple((np.random.uniform() * 1., 0.), np.random.uniform() * 2 * np.pi)
                         for _ in range(self.__n_particles)]
        self.__particles_vel = cuda.to_device(particles_vel)

        particles_type = []
        for n in range(len(N_PARTICLES)):
            particles_type.extend([n for _ in range(N_PARTICLES[n])])
        self.__particles_type = cuda.to_device(particles_type)

        field = np.zeros((FIELD_WIDTH, FIELD_HEIGHT, 3), dtype=float)
        self.__field = cuda.to_device(field)

        self.__gust = (0., 0.)

    @property
    def field(self):
        return self.__field.copy_to_host()

    @property
    def particles(self):
        return self.__particles_pos

    @property
    def params(self):
        return self.__params

    def get_type(self, index):
        return self.__particles_type[index]

    def set_param(self, index, value):
        self.__params[index] = value

    def set_energy(self, vel):
        for i in range(len(self.__particles_vel)):
            length = math.sqrt(self.__particles_vel[i][0] ** 2 + self.__particles_vel[i][1] ** 2)
            self.__particles_vel[i] = (self.__particles_vel[i][0] * vel / length,
                                       self.__particles_vel[i][1] * vel / length)

    def update(self):
        bpg_x = math.ceil(self.__field.shape[0] / TPB[0])
        bpg_y = math.ceil(self.__field.shape[1] / TPB[1])
        if TRAILS:
            decay_trails[(bpg_x, bpg_y), TPB](self.__field)
        else:
            clear_field[(bpg_x, bpg_y), TPB](self.__field)

        self.__gust = rotate_tuple(self.__gust, np.random.normal(scale=0.05) * 2 * np.pi)

        bpg = math.ceil(self.__n_particles / TPB_LIN)
        params = cuda.to_device(self.__params)
        update_particles[bpg, TPB_LIN](
            self.__particles_pos, self.__particles_vel,
            self.__particles_type, self.__field, params,
            self.__gust
        )


def smooth(screen):
    """Apply a gaussian filter to each colour plane"""
    # Get reference pixels for each colour plane and then apply filter
    r = pygame.surfarray.pixels_red(screen)
    gaussian_filter(r, sigma=0, mode="nearest", output=r)
    g = pygame.surfarray.pixels_green(screen)
    gaussian_filter(g, sigma=0, mode="nearest", output=g)
    b = pygame.surfarray.pixels_blue(screen)
    gaussian_filter(b, sigma=0, mode="nearest", output=b)


class Slider:
    def __init__(self, index):
        self.index = index
        self.__x = 10
        self.y = 10 + index * 20 + index // 4 * 122
        self.size = 10
        self.__value = 0
        self.max_value = 100

    @property
    def x(self):
        return self.__x + self.__value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if self.__x <= value < self.__x + self.max_value:
            self.__value = value - self.__x

    def clicked(self, x, y):
        if self.x <= x <= self.x + self.size and self.y <= y <= self.y + self.size:
            return True
        else:
            return False


class View:
    def __init__(self):
        self.__screen = pygame.display.set_mode((SCREEN_WIDTH + 400, SCREEN_HEIGHT))
        self.__clock = pygame.time.Clock()

    def update(self, model, sliders):
        self.__clock.tick(60)

        self.__screen.fill(Color.BLACK)

        """panel = pygame.Surface((400, SCREEN_HEIGHT))

        for slider in sliders:
            rect = pygame.Rect(slider.x, slider.y, slider.size, slider.size)
            pygame.draw.rect(panel, Color.WHITE, rect)

        params = model.params
        for i in range(len(params) // 4):
            pygame.draw.line(panel, Color.RED,
                             (200, i * 200 + 45 + params[i * 4 + 0]),
                             (200 + params[i * 4 + 1], i * 200 + 45))
            pygame.draw.line(panel, Color.GREEN,
                             (200 + params[i * 4 + 1], i * 200 + 45),
                             (200 + params[i * 4 + 1] + (params[i * 4 + 3] - params[i * 4 + 1]) / 2,
                              i * 200 + 45 - params[i * 4 + 2]))
            pygame.draw.line(panel, Color.BLUE,
                             (200 + params[i * 4 + 1] + (params[i * 4 + 3] - params[i * 4 + 1]) / 2,
                              i * 200 + 45 - params[i * 4 + 2]),
                             (200 + params[i * 4 + 3], i * 200 + 45))

        self.__screen.blit(panel, (0, 0))"""

        """image = pygame.surfarray.make_surface(model.field)

        zoomed = pygame.transform.scale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__screen.blit(zoomed, (400, 0))
        # smooth(self.__screen)"""

        field = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        pygame.draw.line(field, Color.RED, (0, 0), (0, SCREEN_HEIGHT))

        for i in range(len(model.particles)):
            if model.get_type(i) == 0:
                pygame.draw.circle(field, Color.RED, model.particles[i], 5)
            elif model.get_type(i) == 1:
                pygame.draw.circle(field, Color.GREEN, model.particles[i], 5)
            else:
                pygame.draw.circle(field, Color.BLUE, model.particles[i], 5)

        self.__screen.blit(field, (400, 0))

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

        # sliders = [Slider(i) for i in range(len(self.__model.params))]
        sliders = []
        clicked_slider = -1

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        file = f'img/trails{screenshot_ctr}.png'
                        self.__view.screenshot(file)
                        screenshot_ctr += 1
                    elif event.key == pygame.K_0:
                        self.__model.set_energy(0)
                    elif event.key == pygame.K_1:
                        self.__model.set_energy(1)
                    elif event.key == pygame.K_2:
                        self.__model.set_energy(2)
                    elif event.key == pygame.K_3:
                        self.__model.set_energy(3)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_LEFT:
                        for slider in sliders:
                            if slider.clicked(*pygame.mouse.get_pos()):
                                clicked_slider = slider.index

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == pygame.BUTTON_LEFT:
                        clicked_slider = -1

            if clicked_slider != -1:
                if clicked_slider % 4 == 1 and sliders[clicked_slider + 2].value < sliders[clicked_slider].value:
                    sliders[clicked_slider + 2].value = pygame.mouse.get_pos()[0]
                    self.__model.set_param(sliders[clicked_slider + 2].index, sliders[clicked_slider + 2].value)
                elif clicked_slider % 4 == 3 and sliders[clicked_slider - 2].value > sliders[clicked_slider].value:
                    sliders[clicked_slider - 2].value = pygame.mouse.get_pos()[0]
                    self.__model.set_param(sliders[clicked_slider - 2].index, sliders[clicked_slider - 2].value)

                sliders[clicked_slider].value = pygame.mouse.get_pos()[0]
                self.__model.set_param(sliders[clicked_slider].index, sliders[clicked_slider].value)

            self.__model.update()

            self.__view.update(self.__model, sliders)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    Controller().run()
