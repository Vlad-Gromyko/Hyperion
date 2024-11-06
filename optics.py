import numpy as np

import cv2
import screeninfo

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from typing import Union, List, Tuple, Iterable
from abc import ABC, abstractmethod

import time
from progress.bar import FillingCirclesBar

import LightPipes as lp

SM = 10 ** -2
MM = 10 ** -3
UM = 10 ** -6
NM = 10 ** -9


class Mesh:
    def __init__(self, width: int = 1920, height: int = 1200, pitch_x: float = 8 * UM, pitch_y: float = 8 * UM):
        self.width = width
        self.height = height

        self.pitch_x = pitch_x
        self.pitch_y = pitch_y

        _x = np.linspace(- width // 2 * pitch_x, width // 2 * pitch_x, width)
        _y = np.linspace(-height // 2 * pitch_y, height // 2 * pitch_y, height)

        self.x, self.y = np.meshgrid(_x, _y)

        self.rho = np.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.atan2(self.y, self.x)


class SLM:
    def __init__(self, mesh: Mesh = Mesh(), gray: int = 255, monitor: int = 1):
        self.mesh = mesh
        self.gray = gray

        self.monitor = monitor

    def translate(self, array: np.ndarray):
        image = self.to_pixels(array)
        screen = screeninfo.get_monitors()[self.monitor]

        cv2.namedWindow('SLM', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('SLM', screen.x - 1, screen.y - 1)
        cv2.setWindowProperty('SLM', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow('SLM', image)
        cv2.waitKey(1)

    def to_pixels(self, array: np.ndarray):
        result = np.asarray(array / 2 / np.pi * self.gray, dtype='uint8')
        return result


class TrapMachine:
    def __init__(self, center: Union[List, Tuple],
                 distance: Union[List, Tuple],
                 dimension: Union[List, Tuple],
                 slm: SLM, wave: float = 850 * NM, focus: float = 100 * MM):

        self.center_x = center[0]
        self.center_y = center[1]

        self.distance_x = distance[0]
        self.distance_y = distance[1]

        self.dimension_x = dimension[0]
        self.dimension_y = dimension[1]

        x_line = [self.center_x - self.distance_x * (self.dimension_x - 1) / 2 + self.distance_x * i for i in
                  range(self.dimension_x)]
        y_line = [self.center_y - self.distance_y * (self.dimension_y - 1) / 2 + self.distance_y * i for i in
                  range(self.dimension_y)]

        x_traps = []
        y_traps = []
        for ix in x_line:
            for iy in y_line:
                x_traps.append(ix)
                y_traps.append(iy)

        self.x_traps = np.asarray(x_traps)
        self.y_traps = np.asarray(y_traps)

        self.num_traps = len(self.x_traps)

        self.slm = slm
        self.wave = wave
        self.focus = focus

    def trap(self, x_trap, y_trap):
        holo = 2 * np.pi / self.wave / self.focus * (x_trap * self.slm.mesh.x + y_trap * self.slm.mesh.y)
        return holo

    def holo_trap(self, x_trap, y_trap):
        return self.trap(x_trap, y_trap) % (2 * np.pi)

    def holo_traps(self, weights=None):
        if weights is None:
            weights = [1 for i in range(self.num_traps)]

        holo = np.zeros((self.slm.mesh.height, self.slm.mesh.width))

        for counter, iw in enumerate(weights):
            holo += self.trap(self.x_traps[counter], self.y_traps[counter])

        return holo % (2 * np.pi)


class Camera:
    def __init__(self, port: int = 0, num_shots: int = 5, width: int = 1280, height: int = 720, pixel: float = 3 * UM):
        self.port = port
        self.num_shots = num_shots

        self.width = width
        self.height = height
        self.pixel = pixel

    def take_shot(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.width)
        shots = []
        for i in range(self.num_shots):
            _, shot = cap.read()
            shots.append(shot)

        shot = np.average(shots, axis=0)
        shot = np.asarray(shot, dtype='uint8')
        shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        shot = np.asarray(shot, dtype='int16')
        cap.release()
        return shot


class Simulator:
    def __init__(self, ):
        pass


class TrapVision:
    def __init__(self, camera: Camera, trap_machine: TrapMachine, slm: SLM, search_radius=5):
        self.camera = camera
        self.trap_machine = trap_machine
        self.slm = slm

        self.registered_x = []
        self.registered_y = []

        self.search_radius = search_radius

        self.back = None
        self.to_show = None

    def register(self):
        bar = FillingCirclesBar('Регистрация Ловушек', max=self.trap_machine.num_traps)
        back_holo = self.trap_machine.holo_trap(0, 2000 * UM)
        self.slm.translate(back_holo)

        self.back = self.camera.take_shot()
        self.to_show = self.back

        for i in range(self.trap_machine.num_traps):
            x_trap = self.trap_machine.x_traps[i]
            y_trap = self.trap_machine.y_traps[i]
            holo = self.trap_machine.holo_trap(x_trap, y_trap)

            self.slm.translate(holo)
            shot = self.camera.take_shot()

            x, y = self.find_trap(np.abs(self.back - shot))
            self.registered_x.append(x)
            self.registered_y.append(y)
            bar.next()

        bar.finish()

    def show_registered(self):
        show = np.asarray(self.to_show, dtype='uint8')
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)
        for i in range(self.trap_machine.num_traps):
            show = cv2.circle(show, (self.registered_x[i], self.registered_y[i]), self.search_radius, (0, 255, 0), 1)
        cv2.imshow('Registered Traps', show)
        cv2.waitKey(1)

    def check_intensities(self, shot):
        intensities = []
        for i in range(self.trap_machine.num_traps):
            value = self.intensity(self.registered_x[i], self.registered_y[i], shot)
            intensities.append(value)

        return intensities

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
        height, width = np.shape(array)
        mask = np.zeros_like(array)

        for iy in range(height):
            for ix in range(width):
                if (ix - x) ** 2 + (iy - y) ** 2 <= self.search_radius ** 2:
                    mask[iy, ix] = 1

        return mask

    def intensity(self, x, y, shot):
        mask = self.masked(x, y, shot)
        shot *= mask
        self.to_show = self.to_show + shot

        return np.max(shot)


class TrapSimulator(TrapVision):
    def __init__(self, camera: Camera, trap_machine: TrapMachine, slm: SLM, search_radius=5, gauss_waist=1 * MM):
        super().__init__(camera, trap_machine, slm, search_radius)
        self.gauss_waist = gauss_waist

        self.slm_grid_dim = max(self.slm.mesh.width, self.slm.mesh.height)
        self.slm_grid_size = self.slm_grid_dim * self.slm.mesh.pitch_x

        self.camera_grid_dim = max(self.camera.width, self.camera.height)
        self.camera_grid_size = self.camera_grid_dim * self.camera.pixel

    def propagate(self, holo):
        field = lp.Begin(self.slm_grid_size, self.trap_machine.wave,
                         self.slm_grid_dim)
        field = lp.GaussBeam(field, self.gauss_waist)

        field = lp.SubPhase(field, holo)
        field = lp.Lens(field, self.trap_machine.focus)
        field = lp.Forvard(field, self.trap_machine.focus)

        field = lp.Interpol(field, self.camera_grid_size,
                            self.camera_grid_dim)
        result = lp.Intensity(field)
        return np.asarray(result / np.max(result) * 255, dtype='int16')

    def register(self):
        bar = FillingCirclesBar('Регистрация Ловушек', max=self.trap_machine.num_traps)
        self.back = np.zeros((self.camera_grid_dim, self.camera_grid_dim))
        self.to_show = self.back
        for i in range(self.trap_machine.num_traps):
            x_trap = self.trap_machine.x_traps[i]
            y_trap = self.trap_machine.y_traps[i]
            holo = self.trap_machine.holo_trap(x_trap, y_trap)

            holo = self.holo_box(holo)
            shot = self.propagate(holo)

            x, y = self.find_trap(np.abs(shot))
            self.registered_x.append(x)
            self.registered_y.append(y)
            bar.next()
        bar.finish()

    @staticmethod
    def holo_box(holo):
        rows, cols = holo.shape

        size = max(rows, cols)

        square_array = np.zeros((size, size), dtype=holo.dtype)

        row_offset = (size - rows) // 2
        col_offset = (size - cols) // 2

        square_array[row_offset:row_offset + rows, col_offset:col_offset + cols] = holo

        return square_array


class Algorithm(ABC):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine, trap_vision: TrapVision, iterations: int):
        self.slm = slm
        self.camera = camera
        self.trap_machine = trap_machine
        self.trap_vision = trap_vision

        self.iterations = iterations

    @abstractmethod
    def run(self):
        pass

    def on_iteration(self):
        pass

    def on_start(self):
        pass

    def on_end(self):
        pass


class GeneticAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine, trap_vision: TrapVision, iterations: int):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

    def run(self):
        pass


if __name__ == '__main__':
    _slm = SLM()
    _camera = Camera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (3, 3), _slm)

    sim = TrapSimulator(_camera, _tr, _slm)
    sim.register()
    sim.show_registered()

    print(sim.registered_x, sim.registered_y)

    cv2.waitKey(0)
