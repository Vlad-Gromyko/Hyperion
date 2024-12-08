import collections
import copy
import time
from turtledemo.penrose import start

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lab import *

import numba


class GSplusWeight(Experiment):
    def __init__(self, slm: SLM = SLM(), vision: Camera | VirtualCamera = Camera(), wave=850 * NM, focus=100 * MM,
                 search_radius=20):
        super().__init__(slm, vision, wave, focus, search_radius)
        self.u_history = []
        self.weights_history = []
        self.intensities_history = []
        self.gradient_history = []

        self.best_weights = []
        self.best_uniformity = []
        self.best_intensities = []

        self.d_history = []
        self.best_d_history = []

        self.solution = []

        self.fig, self.axs = None, None

        self.stag = 0

    def iteration(self, weights):
        starter = np.random.uniform(0, 2 * np.pi, (self.slm.height, self.slm.width))
        hota = lambda solution: mega_HOTA(self.x_traps, self.y_traps, self.slm.x, self.slm.y, self.wave,
                                          self.focus, solution, starter, 20)

        holo = hota(weights)
        self.to_slm(holo)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u = self.uniformity(values / self.design)
        d = np.sum(np.abs(values - np.asarray(self.design) / np.max(self.design)))

        if (len(self.best_uniformity) == 0) or (u > self.best_uniformity[-1]):
            self.best_uniformity.append(u)
            self.best_weights.append(weights)
            self.best_d_history.append(d)
            print('best')
            self.stag = 0
        else:
            self.stag += 1
            if self.stag == 20:
                self.stag = 0
                print('stagnation')

        self.weights_history.append(weights)
        self.intensities_history.append(values)
        self.u_history.append(u)

        self.d_history.append(d)
        self.gradient_history.append(np.zeros(self.num_traps))
        plt.title(f'{len(self.u_history)},   u = {u}')
        self.axs[0].clear()
        self.axs[0].plot([i for i in range(len(self.u_history))], self.u_history)

        self.axs[1].clear()
        self.axs[1].bar([i for i in range(len(values))], values)

        self.axs[2].clear()
        self.axs[2].bar([i for i in range(self.num_traps)], weights)

        self.axs[3].clear()
        self.axs[3].plot([i for i in range(len(self.d_history))], self.d_history)

        self.axs[4].clear()
        self.axs[4].bar([i for i in range(len(self.design))], self.design)

        self.axs[0].set_ylabel('Uniformity')
        self.axs[1].set_ylabel('Intensities')
        self.axs[2].set_ylabel('Weights')
        self.axs[3].set_ylabel('Deviation')
        self.axs[4].set_ylabel('Target')

        plt.title(f'{len(self.u_history)}, u = {self.u_history[-1]}')

        plt.draw()

        plt.gcf().canvas.flush_events()

        # image = ImageGrab.grab()
        # image.save(f'4x4/{len(self.u_history)}.png')

    def run(self, iterations):
        np.random.seed(2)
        self.fig, self.axs = plt.subplots(5, 1, layout='constrained')
        # self.design = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.design = np.asarray(self.design)
        weights = self.design
        # weights = np.random.uniform(1, 1, self.num_traps)
        self.iteration(weights)

        values = self.intensities_history[-1]

        velocity = 1
        thresh = 1
        start_thresh = 1

        for k in range(iterations):
            values = self.intensities_history[-1]

            avg = np.average(values)
            u = self.u_history[-1]
            d = self.d_history[-1]

            davg = np.average(values, weights=1 / self.design)

            thresh = start_thresh if self.stag == 0 else thresh * 2
            # thresh = start_thresh if d <= self.best_d_history[-1] else thresh * 2
            print(velocity / thresh)
            weights = self.best_weights[-1]
            weights = weights + velocity * (self.design / np.max(self.design) - values) / thresh

            self.iteration(weights)

    @staticmethod
    def calculate_k(x, y):
        mx = x - x.mean()
        my = y - y.mean()
        return sum(mx * my) / sum(mx ** 2)


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


if __name__ == '__main__':
    plt.ion()
    exp = GSplusWeight(vision=VirtualCamera())
    # exp.zernike_fit(30)
    exp.add_array(0 * UM, 0, 85 * UM, 85 * UM, 4, 4, func=lambda i: i//4 + 1)
    # exp.add_image('../images/ring.jpg', size_x=10, size_y=10, d_x=60*UM, d_y=60*UM)
    # exp.add_circle_array(0 * UM, 0, 300 * UM, 15)
    # exp.add_circle_array(0 * UM, 0, 150 * UM, 5, func=lambda i: i+1)

    # print('Угол наклона координатной сетки  ::  ', exp.angle_correct(500 * UM, 800 * UM))

    # exp.apply_angle_correction()

    # exp.show_trap_map()

    exp.register_traps()

    exp.run(300)
    plt.ioff()
    plt.show()
