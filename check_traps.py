import numpy as np

import matplotlib.pyplot as plt
import numba

import cv2
import screeninfo
import time


def plot2d(array):
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap='hot')

    fig.colorbar(im, ax=ax, )

    plt.show()


def back_HOTA(x_list, y_list, x_mesh, y_mesh, wave, focus, user_weights, initial_phase, iterations):
    num_traps = len(user_weights)
    v_list = np.zeros_like(user_weights, dtype=np.complex128)
    area = np.shape(initial_phase)[0] * np.shape(initial_phase)[1]
    phase = np.zeros_like(initial_phase)

    w_list = np.ones(num_traps)

    lattice = 2 * np.pi / wave / focus

    for i in range(num_traps):
        trap = (lattice * (x_list[i] * x_mesh + y_list[i] * y_mesh)) % (2 * np.pi)
        v_list[i] = 1 / area * np.sum(np.exp(1j * (initial_phase - trap)))

    anti_user_weights = 1 / user_weights

    for k in range(iterations):
        w_list_before = w_list
        avg = np.average(np.abs(v_list), weights=anti_user_weights)

        w_list = avg / np.abs(v_list) * user_weights * w_list_before

        summ = np.zeros_like(initial_phase, dtype=np.complex128)
        for ip in range(num_traps):
            trap = (lattice * (x_list[ip] * x_mesh + y_list[ip] * y_mesh)) % (2 * np.pi)
            summ = summ + np.exp(1j * trap) * user_weights[ip] * v_list[ip] * w_list[ip] / np.abs(
                v_list[ip])
        phase = np.angle(summ)

        for iv in range(num_traps):
            trap = (lattice * (x_list[iv] * x_mesh + y_list[iv] * y_mesh)) % (2 * np.pi)
            v_list[iv] = 1 / area * np.sum(np.exp(1j * (phase - trap)))
    return phase


def to_pixels(array):
    result = np.asarray(array / 2 / np.pi * 255, dtype='uint8')
    return result


def to_slm(image):
    image = to_pixels(image)
    screen_id = 1
    screen = screeninfo.get_monitors()[screen_id]

    cv2.namedWindow('SLM', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('SLM', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('SLM', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    cv2.imshow('SLM', image)
    cv2.waitKey(1)


def take_shot():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    _, shot = cap.read()
    shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    shot = np.asarray(shot, dtype='int16')
    cap.release()
    return shot


def find_center(image):
    res_y, res_x = np.shape(image)
    ax = np.linspace(0, res_x, res_x)
    ay = np.linspace(0, res_y, res_y)

    ax, ay = np.meshgrid(ax, ay)

    x_c = int(np.sum(ax * image) / np.sum(image))
    y_c = int(np.sum(ay * image) / np.sum(image))

    return x_c, y_c


def find_trap(array):
    spot = np.max(array)
    mask = np.where(array == spot, array, 0)
    x, y = find_center(mask)
    return x, y


def calc_holo(x, y):
    _x = np.linspace(- WIDTH // 2 * PIXEL, WIDTH // 2 * PIXEL, WIDTH)
    _y = np.linspace(-HEIGHT // 2 * PIXEL, HEIGHT // 2 * PIXEL, HEIGHT)

    _x, _y = np.meshgrid(_x, _y)

    lattice = 2 * np.pi / WAVE / FOCUS * (x * _x + y * _y)
    return lattice % (2 * np.pi)


def back():
    holo = calc_holo(0, 2000 * um)
    to_slm(holo)
    return take_shot()


def register_traps(x_list, y_list, back_shot):
    x_spots = []
    y_spots = []

    for i in range(len(x_list)):
        holo = calc_holo(x_list[i], y_list[i])

        to_slm(holo)

        shot = take_shot()

        x, y = find_trap(np.abs(shot - back_shot))

        x_spots.append(x)
        y_spots.append(y)


    return x_spots, y_spots

def draw_trap(c_x, c_y, array):

    array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    array = cv2.circle(array, (c_x, c_y), RADIUS, (0, 255, 0), 1)
    b, g, r = cv2.split(array)

    b[int(c_y)] = 0
    b[:, int(c_x)] = 0

    g[int(c_y)] = 255
    g[:, int(c_x)] = 255

    r[int(c_y)] = 0
    r[:, int(c_x)] = 0

    array =  cv2.merge([b, g, r])

    return array

def draw_traps(c_x_list, c_y_list, array):

    draw = np.zeros_like(array)

    for i in range(len(c_x_list)):
        draw = draw_trap(c_x_list[i], c_y_list, draw)

    cv2.imshow('TRAPS', draw)






RADIUS = 3

mm = 10 ** -3
um = 10 ** -6
nm = 10 ** -9

X_D = 120 * um
Y_D = 120 * um

X_C = 0 * um
Y_C = 0

X_N = 2
Y_N = 2

WIDTH = 1920
HEIGHT = 1200
PIXEL = 8 * um

FOCUS = 100 * mm
WAVE = 850 * nm

ITERATIONS = 30

if __name__ == '__main__':
    x_line = [X_C - X_D * (X_N - 1) / 2 + X_D * i for i in range(X_N)]
    y_line = [Y_C - Y_D * (Y_N - 1) / 2 + Y_D * i for i in range(Y_N)]

    x_traps = []
    y_traps = []
    for ix in x_line:
        for iy in y_line:
            x_traps.append(ix)
            y_traps.append(iy)

    x_traps = np.asarray(x_traps)
    y_traps = np.asarray(y_traps)

    background = back()

    x_reg, y_reg = register_traps(x_traps, y_traps, background)

    users = np.ones_like(x_traps)

    starter = np.zeros((HEIGHT, WIDTH))

    _x = np.linspace(- WIDTH // 2 * PIXEL, WIDTH // 2 * PIXEL, WIDTH)
    _y = np.linspace(-HEIGHT // 2 * PIXEL, HEIGHT // 2 * PIXEL, HEIGHT)
    _x, _y = np.meshgrid(_x, _y)

    #mega = back_HOTA(x_traps, y_traps, _x, _y, WAVE, FOCUS, users, starter, ITERATIONS)

