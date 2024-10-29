import numpy as np
from functools import lru_cache

import matplotlib.pyplot as plt
import numba

import time


def plot2d(array):
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap='hot')

    fig.colorbar(im, ax=ax, )

    plt.show()



@numba.njit(fastmath=True, parallel=True)
def mega_HOTA(x_list, y_list, x_mesh, y_mesh, user_weights, initial_phase, iterations):
    num_traps = len(user_weights)
    v_list = np.zeros_like(user_weights, dtype=np.complex128)
    area = np.shape(initial_phase)[0] * np.shape(initial_phase)[1]
    phase = np.zeros_like(initial_phase)

    w_list = np.ones(num_traps)

    for i in range(num_traps):
        trap = (x_list[i] * x_mesh + y_list[i] * y_mesh) % (2 * np.pi)
        v_list[i] = 1 / area * np.sum(np.exp(1j * (initial_phase - trap)))

    anti_user_weights = 1 / user_weights

    for k in range(iterations):
        w_list_before = w_list
        avg = np.average(np.abs(v_list), weights=anti_user_weights)

        w_list = avg / np.abs(v_list) * user_weights * w_list_before

        summ = np.zeros_like(initial_phase, dtype=np.complex128)
        for ip in range(num_traps):
            trap = (x_list[ip] * x_mesh + y_list[ip] * y_mesh) % (2 * np.pi)
            summ = summ + np.exp(1j * trap) * user_weights[ip] * v_list[ip] * w_list[ip] / np.abs(
                v_list[ip])
        phase = np.angle(summ)

        for iv in range(num_traps):
            trap = (x_list[iv] * x_mesh + y_list[iv] * y_mesh) % (2 * np.pi)
            v_list[iv] = 1 / area * np.sum(np.exp(1j * (phase - trap)))
    return phase


mm = 10 ** -3
um = 10 ** -6
nm = 10 ** -9

X_D = 120 * um
Y_D = 120 * um

X_C = 1000 * um
Y_C = 0

X_N = 2
Y_N = 3

WIDTH = 1920
HEIGHT = 1200
PIXEL = 8 * um

FOCUS = 100 * mm
WAVE = 850 * nm

ITERATIONS = 30

if __name__ == '__main__':
    x_line = [X_C - X_D * (X_N - 1) / 2 + X_D * i for i in range(X_N)]
    y_line = [Y_C - Y_D * (Y_N - 1) / 2 + Y_D * i for i in range(Y_N)]

    x_line = np.asarray(x_line)
    y_line = np.asarray(y_line)

    users = np.ones_like(x_line)

    starter = np.zeros((HEIGHT, WIDTH))

    _x = np.linspace(- WIDTH // 2 * PIXEL, WIDTH // 2 * PIXEL, WIDTH)
    _y = np.linspace(-HEIGHT // 2 * PIXEL, HEIGHT // 2 * PIXEL, HEIGHT)
    _x, _y = np.meshgrid(_x, _y)

    lattice = 2 * np.pi / WAVE / FOCUS

    _x, _y = _x * lattice, _y * lattice

    start = time.time()
    mega = mega_HOTA(x_line, y_line, _x, _y, users, starter, ITERATIONS)

    t1 = time.time()
    print('END', t1 - start)

    plot2d(mega)
