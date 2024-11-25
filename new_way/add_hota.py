import collections
import copy
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lab import *

import numba


@numba.njit(fastmath=True, parallel=True)
def gs(x_traps, y_traps, wave, focus, x_mesh, y_mesh, starter, solution, iterations):
    num_traps = len(solution)

    phase = starter

    for k in range(iterations):
        summ_phase = np.zeros_like(starter, dtype='complex128')
        for i in range(num_traps):
            trap = 2 * np.pi / wave / focus * (x_mesh * x_traps[i] + y_mesh * y_traps[i])
            solution[i] = np.angle(np.sum(np.exp(-1j * (phase - trap))))
            summ_phase = summ_phase + 1 / num_traps * np.exp(-1j * (trap + solution[i]))
        phase = np.angle(summ_phase)

    return phase, solution


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

        self.solution = []

        self.fig, self.axs = plt.subplots(3, 1, layout='constrained')

    def iteration(self, weights):

        starter = np.random.uniform(0, 2 * np.pi, (self.slm.height, self.slm.width))
        hota = lambda solution: mega_HOTA(self.x_traps, self.y_traps, self.slm.x, self.slm.y, self.wave,
                                          self.focus, solution, starter, 10)

        holo = hota(weights)
        self.to_slm(holo)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u = self.uniformity(values)

        self.weights_history.append(weights)
        self.intensities_history.append(values)
        self.u_history.append(u)
        self.gradient_history.append(np.zeros(self.num_traps))
        plt.title(f'{len(self.u_history)},   u = {u}')
        self.axs[0].clear()
        self.axs[0].plot([i for i in range(len(self.u_history))], self.u_history)

        self.axs[1].clear()
        self.axs[1].bar([i for i in range(len(values))], values)

        self.axs[2].clear()
        self.axs[2].bar([i for i in range(self.num_traps)], weights)

        self.axs[0].set_ylabel('Uniformity')
        self.axs[1].set_ylabel('Intensities')
        self.axs[2].set_ylabel('Weights')

        plt.title(f'{len(self.u_history)}, u = {self.u_history[-1]}')

        plt.draw()

        plt.gcf().canvas.flush_events()

    def run(self, iterations):
        np.random.seed(2)

        weights = np.abs(np.random.standard_normal(self.num_traps) +1)
        self.iteration(weights)

        velocity = 1



        for k in range(iterations):
            values = self.intensities_history[-1]
            avg = np.average(values)
            u = self.u_history[-1]

            thresh = min(k + 1, 100)


            weights = weights + velocity * np.exp(np.max(values) - values) / thresh
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

    exp.add_array(0 * UM, 0, 160 * UM, 160 * UM, 5, 5)
    # exp.add_circle_array(800 * UM, 0, 300 * UM, 15)
    # exp.add_circle_array(800 * UM, 0, 150 * UM, 5)

    # print('Угол наклона координатной сетки  ::  ', exp.angle_correct(500 * UM, 800 * UM))

    # exp.apply_angle_correction()

    # exp.show_trap_map()

    exp.register_traps()

    exp.run(300)
    plt.ioff()
    plt.show()
