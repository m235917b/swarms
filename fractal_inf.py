import colorsys

import mpmath
from numba import njit
import numpy as np
import pygame

SCREEN_WIDTH = 150
SCREEN_HEIGHT = 150

SCREENSHOT_PATH = 'img/screenshot'

N_ITERATIONS = 10000
DIV_RADIUS = 2

COLORS = np.asarray([tuple(c * 255 for c in colorsys.hsv_to_rgb(i / (2 * np.pi), 1, 1)) for i in range(N_ITERATIONS)])


class Color:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    RED = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])


def mandelbrot(r, i, div_radius=2, iterations=1000):
    c = mpmath.mpc(r, i)
    z = mpmath.mpc(0., 0.)

    for i in range(iterations):
        z = z * z + c

        if z.real * z.real + z.imag * z.imag > 4:
            return i

    return -1


@njit
def mandelbrot_f(r, i, div_radius=2, iterations=1000):
    c = complex(r, i)
    z = complex(0., 0.)

    for i in range(iterations):
        z = z * z + c

        if abs(z) > div_radius:
            return i

    return -1


# @njit
def calculate_fractal(field, zoom, pos_x, pos_y, f):
    for x in range(field.shape[0]):
        print(x)
        for y in range(field.shape[1]):
            iters = f(
                (x - mpmath.mpf(field.shape[0] / 2)) / zoom - pos_x,
                (y - mpmath.mpf(field.shape[0] / 2)) / zoom - pos_y,
                DIV_RADIUS, N_ITERATIONS
            )
            color = np.array([0., 0., 0.]) if iters == -1 else COLORS[iters]
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
        self.__field = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=int)
        self.__view = View()
        self.__zoom = mpmath.mpf(1.4411518807585587e+17)
        self.__pos_x, self.__pos_y = mpmath.mpf(0.37514273176922147), mpmath.mpf(-0.6594062123991694)

        mpmath.mp.prec += 100

        pygame.init()

        self.calculate_fractal()

    def calculate_fractal(self):
        calculate_fractal(self.__field, self.__zoom, self.__pos_x, self.__pos_y, mandelbrot)

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
                        self.__pos_x -= (x - mpmath.mpf(SCREEN_WIDTH / 2)) / self.__zoom
                        self.__pos_y -= (y - mpmath.mpf(SCREEN_HEIGHT / 2)) / self.__zoom
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
