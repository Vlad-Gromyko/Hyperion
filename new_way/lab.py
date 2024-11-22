from abc import ABC, abstractmethod

import numpy as np
import numba

import screeninfo
import cv2

import matplotlib.pyplot as plt

SM = 10 ** -2
MM = 10 ** -3
UM = 10 ** -6
NM = 10 ** -9


class SLM:
    def __init__(self, width=1920, height=1200, pixel=8 * UM, gray=255):
        self.gray = gray
        self.width = width
        self.height = height

        self.pixel_x = pixel
        self.pixel_y = pixel

        _x = np.linspace(- width // 2 * pixel, width // 2 * pixel, width)
        _y = np.linspace(-height // 2 * pixel, height // 2 * pixel, height)

        self.x, self.y = np.meshgrid(_x, _y)

        self.rho = np.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.arctan2(self.y, self.x)

    def translate(self, array: np.ndarray):
        image = self.to_pixels(array)
        screen = screeninfo.get_monitors()[1]

        cv2.namedWindow('SLM', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('SLM', screen.x - 1, screen.y - 1)
        cv2.setWindowProperty('SLM', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow('SLM', image)
        cv2.waitKey(1)

    def to_pixels(self, array: np.ndarray):
        result = np.asarray(array / 2 / np.pi * self.gray, dtype='uint8')
        return result


class Vision(ABC):
    def __init__(self, width: int = 1280, height: int = 720, pixel: float = 3 * UM):
        self.width = width
        self.height = height

        self.pixel = pixel

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.width)

    @abstractmethod
    def take_shot(self):
        pass


class Camera(Vision):
    def take_shot(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.width)
        shots = []
        for i in range(15):
            _, shot = cap.read()
            shots.append(shot)

        shot = np.average(shots, axis=0)
        shot = np.asarray(shot, dtype='uint8')
        shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        shot = np.asarray(shot, dtype='int16')
        cap.release()
        return shot


class Experiment:
    def __init__(self, slm: SLM = SLM(), vision: Camera = Camera(), wave=850 * NM, focus=100 * MM, search_radius=20):
        self.slm = slm

        self.wave = wave
        self.focus = focus

        self.x_traps = []
        self.y_traps = []

        self.vision = vision

        self.registered_x = []
        self.registered_y = []

        self.num_traps = 0

        self.back = np.zeros((slm.height, slm.width))

        self.search_radius = search_radius

        self.correction_angle = 0
        self.counter = 0

        self.register_traps()
        self.x_traps = []
        self.y_traps = []

    def show_trap_map(self):
        plt.style.use('dark_background')

        plt.scatter(self.x_traps, self.y_traps)

        plt.show()

    def trap(self, x_trap, y_trap):
        holo = 2 * np.pi / self.wave / self.focus * (x_trap * self.slm.x + y_trap * self.slm.y)
        return holo

    def holo_trap(self, x_trap, y_trap):
        return self.trap(x_trap, y_trap) % (2 * np.pi)

    def add_trap(self, x, y):
        self.x_traps.append(x)
        self.y_traps.append(y)
        self.num_traps += 1

    def add_array(self, c_x, c_y, d_x, d_y, n_x, n_y):
        x_line = [c_x - d_x * (n_x - 1) / 2 + d_x * i for i in
                  range(n_x)]
        y_line = [c_y - d_y * (n_y - 1) / 2 + d_y * i for i in
                  range(n_y)]

        for ix in x_line:
            for iy in y_line:
                self.add_trap(ix, iy)

    def add_circle_array(self, c_x, c_y, radius, num):
        angle = 2 * np.pi / num

        for k in range(num):
            self.add_trap(radius * np.cos(angle) + c_x, radius * np.sin(angle) + c_y)
            angle += 2 * np.pi / num

    def check_intensities(self, shot):
        values = []
        for i in range(self.num_traps):
            value = self.intensity(self.registered_x[i], self.registered_y[i], shot)
            self.draw_area(self.registered_y[i], self.registered_x[i], shot)
            values.append(value)

        return np.asarray(values) / np.max(values)

    def draw_area(self, x, y, shot):
        shot = shot / np.max(shot) * 255
        shot = np.asarray(shot, dtype='uint8')
        shot = cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR)

        show = cv2.rectangle(shot, (x - self.search_radius, y - self.search_radius),
                             (x + self.search_radius, y + self.search_radius), (0, 255, 0), 1)

        # show = cv2.resize(show, (500, 500))
        cv2.imshow('Registered Traps', show)
        cv2.waitKey(1)

    @staticmethod
    def find_center(image):
        res_y, res_x = np.shape(image)
        ax = np.linspace(0, res_x, res_x)
        ay = np.linspace(0, res_y, res_y)

        ax, ay = np.meshgrid(ax, ay)

        x_c = int(np.sum(ax * image) / np.sum(image))
        y_c = int(np.sum(ay * image) / np.sum(image))
        return x_c, y_c

    def find_trap(self, array):
        spot = np.max(array)
        mask = np.where(array == spot, array, 0)
        x, y = self.find_center(mask)
        return x, y

    def masked(self, x, y, array):
        mask = np.zeros_like(array)

        # for iy in range(height):
        # for ix in range(width):
        # if (ix - x) ** 2 + (iy - y) ** 2 <= self.search_radius ** 2:
        # mask[iy, ix] = 1

        mask[x - self.search_radius: x + self.search_radius, y - self.search_radius: y + self.search_radius] = 1

        return mask

    def intensity(self, x, y, shot):
        mask = self.masked(x, y, shot)
        return np.max(shot * mask)

    def register_traps(self):

        self.back = self.vision.take_shot()

        for i in range(self.num_traps):
            x_trap = self.x_traps[i]
            y_trap = self.y_traps[i]
            holo = self.holo_trap(x_trap, y_trap)

            self.slm.translate(holo)
            shot = self.vision.take_shot()

            y, x = self.find_trap(np.abs(self.back - shot))
            self.registered_x.append(x)
            self.registered_y.append(y)
            print('REG ', i + 1, 'X = ', x, 'Y = ', y)
            # bar.next()

        self.x_traps = np.asarray(self.x_traps)
        self.y_traps = np.asarray(self.y_traps)

        # plt.clf()

        # plt.imshow(shot)

        # plt.draw()
        # plt.gcf().canvas.flush_events()

        # bar.finish()

    def run(self, iterations):
        pass

    def holo_weights_and_phases(self, weights, phases):
        weights = np.asarray(weights)
        phases = np.asarray(phases)
        return numba_w_p_kernel(weights, phases, self.x_traps, self.y_traps, self.wave, self.focus, self.slm.width,
                                self.slm.height, self.slm.x, self.slm.y)

    def angle_correct(self, delta_x, c_x=0 * UM):
        holo = self.holo_trap(c_x - delta_x / 2, 0)

        self.slm.translate(holo)
        shot = self.vision.take_shot()

        y_left, x_left = self.find_trap(np.abs(self.back - shot))

        holo = self.holo_trap(c_x + delta_x / 2, 0)

        self.slm.translate(holo)
        shot = self.vision.take_shot()

        y_right, x_right = self.find_trap(np.abs(self.back - shot))

        self.correction_angle = np.arctan2(y_left - y_right, x_left - x_left)
        return self.correction_angle

    def apply_angle_correction(self):
        for i in range(len(self.x_traps)):
            self.x_traps[i], self.y_traps[i] = self.angle(self.x_traps[i], self.y_traps[i], -self.correction_angle)

    @staticmethod
    def angle(x, y, theta):
        x_ = x * np.cos(theta) - y * np.sin(theta)
        y_ = x * np.sin(theta) + y * np.cos(theta)
        return x_, y_

    @staticmethod
    def uniformity(values):
        return 1 - (np.max(values) - np.min(values)) / (np.max(values) + np.min(values))


@numba.njit(fastmath=True, parallel=True)
def numba_w_p_kernel(weights, phases, x_traps, y_traps, wave, focus, width, height, x_mesh, y_mesh):
    holo = np.zeros((height, width), dtype='complex128')

    for counter, iw in enumerate(weights):
        trap = 2 * np.pi / wave / focus * (x_traps[counter] * x_mesh + y_traps[counter] * y_mesh)
        holo += np.exp(-1j * (trap - phases[counter])) * iw
    holo = np.angle(holo)
    return holo + np.pi
