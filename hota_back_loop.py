import cv2
import screeninfo

import numpy as np

import matplotlib.pyplot as plt
import time

mm = 10 ** -3
um = 10 ** -6
nm = 10 ** -9

X_D = 120 * um
Y_D = 120 * um
Z_D = 120 * um

X_C = 1000 * um
Y_C = 0
Z_C = 0

X_N = 5
Y_N = 5
Z_N = 1

WIDTH = 1920
HEIGHT = 1200
PIXEL = 8 * um

FOCUS = 100 * mm
WAVE = 850 * nm


def calc_holo(x, y, z=0):
    _x = np.linspace(- WIDTH // 2 * PIXEL, WIDTH // 2 * PIXEL, WIDTH)
    _y = np.linspace(-HEIGHT // 2 * PIXEL, HEIGHT // 2 * PIXEL, HEIGHT)

    _x, _y = np.meshgrid(_x, _y)

    sphere = np.pi * z / WAVE / FOCUS / FOCUS * (_x * _x + _y * _y)
    lattice = 2 * np.pi / WAVE / FOCUS * (x * _x + y * _y)
    return (sphere + lattice) % (2 * np.pi)


def to_pixels(array):
    result = np.asarray(array / 2 / np.pi * 255, dtype='uint8')
    return result


def find_center(image):
    res_y, res_x = np.shape(image)
    ax = np.linspace(0, res_x, res_x)
    ay = np.linspace(0, res_y, res_y)

    ax, ay = np.meshgrid(ax, ay)

    x_c = int(np.sum(ax * image) / np.sum(image))
    y_c = int(np.sum(ay * image) / np.sum(image))

    return x_c, y_c


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



def plot2d(array):
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap='hot')

    fig.colorbar(im, ax=ax, )

    plt.show()



if __name__ == '__main__':


    shift = calc_holo(0, 2002 * um)
    to_slm(shift)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _, back = cap.read()
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    back = np.asarray(back, dtype='int16')

    plot2d(back)
    cap.release()

    holo = calc_holo(500 * um, 0)
    to_slm(holo)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    _, shot = cap.read()
    shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    shot = np.asarray(shot, dtype='int16')
    plot2d(shot)
    cap.release()

    delta = np.abs(shot - back)

    xc, yc = find_center(delta)

    plot2d(delta)
