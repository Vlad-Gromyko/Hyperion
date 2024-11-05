import numpy as np

import screeninfo
import cv2

MM = 10 ** -3
SM = 10 ** -2
UM = 10 ** -6
NM = 10 ** -9


class Lab:
    def __init__(self, width=1920, height=1200, gray=255, pixel_pitch=8 * UM, wave=850 * NM, focus=100 * UM):
        self.width = width
        self.height = height
        self.gray = gray
        self.pixel_pitch = pixel_pitch
        self.wave = wave
        self.focus = focus

        self.mesh_x, self.mesh_y = self.do_mesh()

        self.back_shot = self.back()





    def do_mesh(self):
        _x = np.linspace(- self.width // 2 * self.pixel_pitch, self.width // 2 * self.pixel_pitch, self.width)
        _y = np.linspace(-self.height // 2 * self.pixel_pitch, self.height // 2 * self.pixel_pitch, self.height)

        return np.meshgrid(_x, _y)

    def holo_trap(self, x_trap, y_trap):
        holo = 2 * np.pi / self.wave / self.focus * (x_trap * self.mesh_x + y_trap * self.mesh_y)
        return holo % (2 * np.pi)

    def holo_traps(self, x_traps, y_traps, weights=None):
        if weights is None:
            weights = [1 for i in range(len(x_traps))]

        holo = np.zeros((self.height, self.width))

        for counter, iw in enumerate(weights):
            holo += self.holo_trap(x_traps[counter], y_traps[counter])

        return holo % (2 * np.pi)

    def to_pixel(self, array):
        result = np.asarray(array / 2 / np.pi * self.gray, dtype='uint8')
        return result

    def to_slm(self, array):
        image = self.to_pixel(array)

        screen_id = 1
        screen = screeninfo.get_monitors()[screen_id]

        cv2.namedWindow('SLM', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('SLM', screen.x - 1, screen.y - 1)
        cv2.setWindowProperty('SLM', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow('SLM', image)
        cv2.waitKey(1)

    def take_shot(self, nums=5):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        shots = []
        for i in range(nums):
            _, shot = cap.read()
            shots.append(shot)

        shot = np.average(shots, axis=0)
        shot = np.asarray(shot, dtype='uint8')
        shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        shot = np.asarray(shot, dtype='int16')
        cap.release()
        return shot

    def back(self):
        return self.take_shot()

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

    def registration(self, x_traps, y_traps):
        x_spots = []
        y_spots = []

        for i in range(x_traps):
            holo = self.holo_trap(x_traps[i], y_traps[i])

            self.to_slm(holo)

            shot = self.take_shot()

            x, y = self.find_trap(np.abs(shot - self.back_shot))

            x_spots.append(x)
            y_spots.append(y)

            print(i + 1)

        return x_spots, y_spots

    @staticmethod
    def masked(x, y, radius, array):
        height, width = np.shape(array)
        mask = np.zeros_like(array)

        for iy in range(height):
            for ix in range(width):
                if (ix - x) ** 2 + (iy - y) ** 2 <= radius ** 2:
                    mask[iy, ix] = 1

        return mask

    def intensity(self, x, y, radius, shot):
        mask = masked(x, y, radius, shot)
        shot *= mask
        show = np.asarray(shot, dtype='uint8')
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)

        show = cv2.circle(show, (x, y), RADIUS, (0, 255, 0), 1)

        cv2.imshow('TRAP', show)
        cv2.waitKey(1)
        return np.max(shot)



class HotaGeneticAlgorithm:
    def __init__(self, x_traps, y_traps, laboratory):
