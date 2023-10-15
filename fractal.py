import math
import colorsys

from numba import cuda
import numpy as np
import pygame

TPB = (16, 16)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

SCREENSHOT_PATH = 'img/screenshot'

N_ITERATIONS = 100000
DIV_RADIUS = 2

COLORS = np.asarray([tuple(c * 255 for c in colorsys.hsv_to_rgb(2 * np.pi * i / N_ITERATIONS, 1, 1))
                     for i in range(N_ITERATIONS)])


class Color:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    RED = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])


@cuda.jit
def mandelbrot(r, i, div_radius=2, iterations=1000):
    c = complex(r, i)
    z = complex(0., 0.)

    for i in range(iterations):
        z = z * z * z * z + c

        if abs(z) > div_radius:
            return i

    return -1


@cuda.jit
def calculate_fractal(field, pos_x, pos_y, zoom, cc):
    x, y = cuda.grid(2)

    if x < field.shape[0] and y < field.shape[1]:
        iters = mandelbrot(
            (x - field.shape[0] / 2) / zoom - pos_x,
            (y - field.shape[0] / 2) / zoom - pos_y,
            DIV_RADIUS, N_ITERATIONS
        )

        if iters == -1:
            field[x, y, 0] = 0
            field[x, y, 1] = 0
            field[x, y, 2] = 0
        else:
            color = cc[iters]
            field[x, y, 0] = color[0]
            field[x, y, 1] = color[1]
            field[x, y, 2] = color[2]


class View:
    def __init__(self):
        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__clock = pygame.time.Clock()

    def update(self, field):
        self.__clock.tick(60)

        self.__screen.fill(Color.BLACK)

        # draw particle field

        image = pygame.surfarray.make_surface(field)

        zoomed = pygame.transform.scale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.__screen.blit(zoomed, (0, 0))

        pygame.display.flip()

    def screenshot(self, file):
        pygame.image.save(self.__screen, file)


class Controller:
    def __init__(self):
        self.__field = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=float)
        self.__field_cuda = cuda.to_device(self.__field)
        self.__view = View()
        self.__zoom = 1.8014398509481984e+16
        self.__pos_x, self.__pos_y = 0.4424886712117647, -0.500232154954534
        self.__colors_cuda = cuda.to_device(COLORS)
        pygame.init()

        self.calculate_fractal()

    def calculate_fractal(self):
        bpg_x = math.ceil(self.__field_cuda.shape[0] / TPB[0])
        bpg_y = math.ceil(self.__field_cuda.shape[1] / TPB[1])
        calculate_fractal[(bpg_x, bpg_y), TPB](
            self.__field_cuda, self.__pos_x, self.__pos_y, self.__zoom, self.__colors_cuda
        )
        self.__field = self.__field_cuda.copy_to_host()

        """field = self.__field_cuda.copy_to_host()
        rgb_field = []

        for row in field:
            r = []
            for col in row:
                c = colorsys.hsv_to_rgb(*col)
                r.append((c[0] * 255, c[1] * 255, c[2] * 255))

            rgb_field.append(r)

        self.__field = np.array(rgb_field)"""

    def run(self):
        running = True
        screenshot_ctr = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        file = f'{SCREENSHOT_PATH}{screenshot_ctr}.png'
                        self.__view.screenshot(file)
                        screenshot_ctr += 1

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_LEFT:
                        x, y = pygame.mouse.get_pos()
                        self.__pos_x -= (x - SCREEN_WIDTH / 2) / self.__zoom
                        self.__pos_y -= (y - SCREEN_HEIGHT / 2) / self.__zoom
                        self.__zoom *= 2.

                        print(self.__pos_x, self.__pos_y, self.__zoom)

                        self.calculate_fractal()
                    elif event.button == pygame.BUTTON_RIGHT:
                        self.__zoom /= 2.

                        self.calculate_fractal()

            self.__view.update(self.__field)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    Controller().run()
