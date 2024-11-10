import numpy as np
import numba

import cv2
import screeninfo

import LightPipes as lp

from abc import ABC, abstractmethod

SM = 10 ** -2
MM = 10 ** -3
UM = 10 ** -6
NM = 10 ** -9


class SLM:
    def __init__(self, width=1920, height=1200, pixel=8 * UM, gray=255, port=1):
        self.width = width
        self.height = height

        self.pixel = pixel

        _x = np.linspace(- width // 2 * pixel, width // 2 * pixel, width)
        _y = np.linspace(-height // 2 * pixel, height // 2 * pixel, height)

        self.x, self.y = np.meshgrid(_x, _y)

        self.rho = np.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.atan2(self.y, self.x)

        self.gray = gray
        self.port = port

    def translate(self, array: np.ndarray):
        image = self.to_pixels(array)
        screen = screeninfo.get_monitors()[self.port]

        cv2.namedWindow('SLM', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('SLM', screen.x - 1, screen.y - 1)
        cv2.setWindowProperty('SLM', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow('SLM', image)
        cv2.waitKey(1)

    def to_pixels(self, array: np.ndarray):
        result = np.asarray(array / 2 / np.pi * self.gray, dtype='uint8')
        return result


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


class HoloFactory:
    def __init__(self, slm: SLM, focus: float, wave: float):
        self.slm = slm
        self.focus = focus
        self.wave = wave

    def holo_trap(self, x: float, y: float):
        holo = 2 * np.pi / self.wave / self.focus * (x * self.slm.x + y * self.slm.y)

        return holo % (2 * np.pi)

    def holo_traps(self, x_traps, y_traps, weights):
        return (mega_HOTA(x_traps, y_traps, self.slm.x, self.slm.y,
                          self.wave, self.focus, weights, np.zeros((self.slm.height, self.slm.width)), 10)
                + np.pi)


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
    return phase


class TrapsExperiment(ABC):
    def __init__(self, slm: SLM = SLM(), camera: Camera = Camera(), focus: float = 100 * MM, wave: float = 850 * NM,
                 search_radius=15, gauss=1 * MM):
        self.slm = slm
        self.camera = camera
        self.wave = wave
        self.focus = focus
        self.gauss = gauss

        self.search_radius = search_radius

        self.holo = HoloFactory(slm, focus, wave)

        self.x = []
        self.y = []

        self.reg_x = []
        self.reg_y = []

        self.back = np.zeros((camera.height, camera.width))

        self.sum_field = np.zeros_like(self.back)

        self.num_traps = 0

    @abstractmethod
    def run(self, iterations):
        pass

    def add_array(self, c_x, c_y, d_x, d_y, n_x, n_y):

        x_line = [c_x - d_x * (n_x - 1) / 2 + d_x * i for i in range(n_x)]
        y_line = [c_y - d_y * (n_y - 1) / 2 + d_y * i for i in range(n_y)]

        x_traps = []
        y_traps = []
        for ix in x_line:
            for iy in y_line:
                x_traps.append(ix)
                y_traps.append(iy)

        for i in range(len(x_traps)):
            self.add_trap(x_traps[i], y_traps[i])

    def add_trap(self, x, y):
        self.x.append(x)
        self.y.append(y)

        self.register_trap(x, y)

    def check_intensities(self, shot):
        values = []
        for i in range(self.num_traps):
            value = self.intensity(self.reg_x[i], self.reg_y[i], shot)
            values.append(value)

        return np.asarray(values) / np.max(values)

    def masked(self, x, y, array):
        height, width = np.shape(array)
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

    @abstractmethod
    def to_slm(self, holo: np.ndarray):
        pass

    @abstractmethod
    def take_shot(self):
        pass

    def register_trap(self, x: float, y: float):
        holo = self.holo.holo_trap(x, y)
        self.to_slm(holo)

        shot = self.take_shot()

        #cv2.imshow('', shot)
        #cv2.waitKey(1)

        fx, fy = self.find_trap(np.abs(shot))
        self.reg_x.append(fy)
        self.reg_y.append(fx)

        self.sum_field = self.sum_field + shot

        self.num_traps += 1
        print('REG  ', self.num_traps, fy, fx)

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

    @staticmethod
    def box(array):
        rows, cols = array.shape

        size = max(rows, cols)

        square_array = np.zeros((size, size), dtype=array.dtype)

        row_offset = (size - rows) // 2
        col_offset = (size - cols) // 2

        square_array[row_offset:row_offset + rows, col_offset:col_offset + cols] = array

        return square_array


class RealTrapExperiment(TrapsExperiment):
    def __init__(self, slm: SLM = SLM(), camera: Camera = Camera(), focus: float = 100 * MM, wave: float = 850 * NM,
                 search_radius=15, gauss=1 * MM):
        super().__init__(slm, camera, focus, wave, search_radius, gauss)

    def to_slm(self, holo: np.ndarray):
        self.slm.translate(holo)

    def take_shot(self):
        return self.camera.take_shot()


class VirtualTrapExperiment(TrapsExperiment):
    def __init__(self, slm: SLM = SLM(), camera: Camera = Camera(), focus: float = 100 * MM, wave: float = 850 * NM,
                 search_radius=5, gauss=1 * MM):
        super().__init__(slm, camera, focus, wave, search_radius, gauss)


        self.back = self.box(self.back)
        self.sum_field = self.box(self.sum_field)


        self.slm_size = max(self.slm.height, self.slm.width)
        self.slm_proxy = np.zeros((self.slm.height, self.slm.width))


        self.camera_size = max(self.camera.height, self.camera.width)

        self.slm_grid_dim = max(self.slm.width, self.slm.height)
        self.slm_grid_size = self.slm_grid_dim * self.slm.pixel

        self.camera_grid_dim = max(self.camera.width, self.camera.height)
        self.camera_grid_size = self.camera_grid_dim * self.camera.pixel / 3


        self.field = lp.Begin(self.slm_grid_size, self.wave,
                              self.slm_grid_dim)

        self.field = lp.GaussBeam(self.field, self.gauss)


    def to_slm(self, holo: np.ndarray):
        self.slm_proxy = self.box(holo)
        self.slm_proxy = self.slm.to_pixels(self.slm_proxy)

        self.slm.translate(holo)


    def take_shot(self):
        field = lp.SubPhase(self.field, self.slm_proxy)
        field = lp.Lens(field, self.focus)
        field = lp.Forvard(field, self.focus)

        field = lp.Interpol(field, self.camera_grid_size,
                            self.camera_grid_dim)
        result = lp.Intensity(field)
        return result

    def resize_to_slm(self, shot):
        image = np.asarray(shot * 255, dtype='uint8')
        image = cv2.resize(image, (self.slm_size // 4, self.slm_size // 4))
        image = np.asarray(image, dtype='float64')
        image = image / np.max(image)

        slm = np.zeros((self.slm_size, self.slm_size), dtype='float64')
        slm[self.slm_size // 2 - self.slm_size // 8: self.slm_size // 2 + self.slm_size // 8,
        self.slm_size // 2 - self.slm_size // 8: self.slm_size // 2 + self.slm_size // 8] = image
