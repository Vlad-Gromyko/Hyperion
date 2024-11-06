import numpy as np

import matplotlib.pyplot as plt
import numba

import cv2
import screeninfo
import time
import copy


@numba.njit(fastmath=True, parallel=True)
def mega_HOTA(x_list, y_list, x_mesh, y_mesh, wave, focus, user_weights, initial_phase, iterations):
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
    return phase, w_list, v_list


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
        w_list_before = copy.deepcopy(w_list)
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
    shots = []
    for i in range(SHOTS):
        _, shot = cap.read()
        shots.append(shot)

    shot = np.average(shots, axis=0)
    shot = np.asarray(shot, dtype='uint8')
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
    shot = take_shot()
    return shot


def draw(x_list, y_list, image):
    image = np.asarray(image, dtype='uint8')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(image)
    for i in range(len(x_list)):
        image = cv2.circle(image, (x_list[i], y_list[i]), RADIUS, (0, 255, 0), 1)

    return image


def intensity(x, y, radius, shot):
    mask = masked(x, y, radius, shot)
    shot *= mask
    show = np.asarray(shot, dtype='uint8')
    show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)

    show = cv2.circle(show, (x, y), RADIUS, (0, 255, 0), 1)

    cv2.imshow('TRAP', show)
    cv2.waitKey(1)
    return np.max(shot)


def masked(x, y, radius, array):
    height, width = np.shape(array)
    mask = np.zeros_like(array)

    for iy in range(height):
        for ix in range(width):
            if (ix - x) ** 2 + (iy - y) ** 2 <= radius ** 2:
                mask[iy, ix] = 1

    return mask


def uniformity(v_list):
    return 1 - (np.max(v_list) - np.min(v_list)) / (np.max(v_list) + np.min(v_list))


def show_us(name, array):
    cv2.imshow(name, array)
    cv2.waitKey(1)


def exp_HOTA(x_list, y_list, x_centers, y_centers, x_mesh, y_mesh, wave, focus, user_weights, initial_phase,
             iterations):
    history = []

    background = back()
    num_traps = len(user_weights)
    v_list = np.zeros_like(user_weights, dtype='complex128')
    area = np.shape(initial_phase)[0] * np.shape(initial_phase)[1]
    phase = np.zeros_like(initial_phase)

    w_list = np.ones(num_traps, dtype='float64')

    lattice = 2 * np.pi / wave / focus

    for i in range(num_traps):
        trap = (lattice * (x_list[i] * x_mesh + y_list[i] * y_mesh)) % (2 * np.pi)
        v_list[i] = 1 / area * np.sum(np.exp(1j * (initial_phase - trap)))

    anti_user_weights = 1 / user_weights

    '''
    for iv in range(num_traps):
        value = np.sqrt(intensity(x_centers[iv], y_centers[iv], RADIUS, np.abs(shot - background))) + 1
        v_list[iv] = value

    v_list = v_list / np.max(np.abs(v_list))
    
    '''

    for k in range(iterations):
        w_list_before = w_list
        avg = np.average(np.abs(v_list), weights=anti_user_weights)


        w_list = avg / np.abs(v_list) * user_weights * w_list_before

        summ = np.zeros_like(initial_phase, dtype=np.complex128)
        for ip in range(num_traps):
            trap = (lattice * (x_list[ip] * x_mesh + y_list[ip] * y_mesh)) % (2 * np.pi)
            summ = summ + np.exp(1j * trap) * user_weights[ip] * w_list[ip]

        phase = np.angle(summ) + np.pi
        to_slm(phase)

        shot = take_shot()

        for iv in range(num_traps):
            value = np.sqrt(intensity(x_centers[iv], y_centers[iv], RADIUS, np.abs(shot - background)))
            v_list[iv] = value

        v_list = v_list + 1000

        v_list = v_list / np.max(np.abs(v_list))

        history.append(uniformity(np.abs((v_list - 1000) ** 2)))

        plt.clf()

        plt.plot([i for i in range(len(history))], history)

        plt.draw()
        plt.gcf().canvas.flush_events()

        print((np.abs(v_list)))
        print('I`m in HOTA', k)
    return phase


def register_traps(x_list, y_list, back_shot):
    x_spots = []
    y_spots = []

    to_show = np.zeros_like(back_shot)
    array = back_shot
    for i in range(len(x_list)):
        holo = calc_holo(x_list[i], y_list[i])

        to_slm(holo)

        shot = take_shot()

        x, y = find_trap(np.abs(shot - back_shot))

        x_spots.append(x)
        y_spots.append(y)

        print(i + 1)

    array = draw(x_spots, y_spots, array)

    # cv2.imshow('YOOOO', array)
    # cv2.waitKey(0)

    return x_spots, y_spots


RADIUS = 10
SHOTS = 5

mm = 10 ** -3
um = 10 ** -6
nm = 10 ** -9

X_D = 100 * um
Y_D = 100 * um

X_C = 1000 * um
Y_C = 0

X_N = 3
Y_N = 3

WIDTH = 1920
HEIGHT = 1200
PIXEL = 8 * um

FOCUS = 100 * mm
WAVE = 850 * nm

ITERATIONS = 300

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

    plt.ion()

    mega = exp_HOTA(x_traps, y_traps, x_reg, y_reg, _x, _y, WAVE, FOCUS, users, starter, ITERATIONS)

    plt.ioff()

    plot2d(take_shot())

    plt.show()
