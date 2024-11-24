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
        holo = self.holo_weights_and_phases(weights, self.solution)
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
        starter = np.random.uniform(0, 2 * np.pi, (self.slm.height, self.slm.width))

        first_sol = np.random.uniform(0, 2 * np.pi, self.num_traps)

        st_time = time.time()
        phase, self.solution = gs(self.x_traps, self.y_traps,
                                  self.wave, self.focus, self.slm.x, self.slm.y, starter, first_sol, 30)
        print('GS Compute-Time          ::  ', time.time() - st_time)
        print()
        print()

        weights = np.random.uniform(size=self.num_traps)
        self.iteration(weights)

        velocity = 1



        for k in range(iterations):
            values = self.intensities_history[-1]
            avg = np.average(values)
            u = self.u_history[-1]

            thresh = min(k + 1, 20)
            gradient = np.exp((avg - values) * velocity/thresh * (1+u))

            weights = weights * gradient ** np.sign(avg - values)
            self.iteration(weights)


    @staticmethod
    def calculate_k(x, y):
        mx = x - x.mean()
        my = y - y.mean()
        return sum(mx * my) / sum(mx ** 2)

    def fool_gradient(self, weights):
        epsilon = 0.01
        holo = self.holo_weights_and_phases(weights - epsilon / 2, self.solution)
        self.to_slm(holo)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u1 = self.uniformity(values)

        holo = self.holo_weights_and_phases(weights + epsilon / 2, self.solution)
        self.to_slm(holo)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u2 = self.uniformity(values)
        return (u2 - u1) / epsilon

    def gradient(self, weights):
        result = []
        for i in range(self.num_traps):
            result.append(self.gradient_kernel(weights, i))

        return np.asarray(result)

    def gradient_kernel(self, weights, k, epsilon=0.1):
        l_weights = copy.deepcopy(weights)
        l_weights[k] = weights[k] - epsilon / 2
        holo = self.holo_weights_and_phases(l_weights, self.solution)
        self.to_slm(holo)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u1 = self.uniformity(values)
        r_weights = copy.deepcopy(weights)
        r_weights[k] = weights[k] + epsilon / 2
        holo = self.holo_weights_and_phases(r_weights, self.solution)
        self.to_slm(holo)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u2 = self.uniformity(values)
        return (u2 - u1) / epsilon


if __name__ == '__main__':
    plt.ion()
    exp = GSplusWeight(vision=VirtualCamera())

    exp.add_array(0 * UM, 0, 160 * UM, 160 * UM, 2, 2)
    # exp.add_circle_array(800 * UM, 0, 300 * UM, 15)
    # exp.add_circle_array(800 * UM, 0, 150 * UM, 5)

    # print('Угол наклона координатной сетки  ::  ', exp.angle_correct(500 * UM, 800 * UM))

    # exp.apply_angle_correction()

    # exp.show_trap_map()

    exp.register_traps()

    exp.run(300)
    plt.ioff()
    plt.show()
