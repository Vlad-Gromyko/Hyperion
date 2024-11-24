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
    def __init__(self, slm: SLM = SLM(), vision: Camera| VirtualCamera = Camera(), wave=850 * NM, focus=100 * MM, search_radius=20):
        super().__init__(slm, vision, wave, focus, search_radius)
        self.u_history = []
        self.weights_history = []

        self.best_solution = []
        self.best_uniformity = []
        self.best_intensities = []

        self.fig, axs = plt.subplots(4, 1, layout='constrained')
        axs[0].set_ylabel('Uniformity')
        axs[1].set_ylabel('Intensities')
        axs[2].set_ylabel('Weights')
        axs[3].set_ylabel('P')


    def run(self, iterations):
        np.random.seed(2)
        starter = np.random.uniform(0, 2 * np.pi, (self.slm.height, self.slm.width))

        first_sol = np.random.uniform(0, 2 * np.pi, self.num_traps)

        st_time = time.time()
        phase, solution = gs(self.x_traps, self.y_traps,
                             self.wave, self.focus, self.slm.x, self.slm.y, starter, first_sol, 30)
        print('GS Compute-Time          ::  ', time.time() - st_time)
        print()
        print()

        phase = phase + np.pi

        self.to_slm(phase)

        shot = self.take_shot()
        values = self.check_intensities(shot)
        u = self.uniformity(values)

        print('After GS  [Intensities]  ::  ', values)
        print('After GS  [Uniformity]   ::  ', u)
        print('After GS  [Phases]       ::  ', solution)

        print('Start BackLoop...')
        u_history = []
        u_history.append(u)
        best = 0
        weights = np.ones(self.num_traps)
        best_sol = []
        p = np.ones_like(weights)
        fig, axs = plt.subplots(4, 1, layout='constrained')
        axs[0].set_ylabel('Uniformity')
        axs[1].set_ylabel('Intensities')
        axs[2].set_ylabel('Weights')
        axs[3].set_ylabel('P')

        plt.draw()

        plt.gcf().canvas.flush_events()

        for k in range(iterations):

            avg = np.average(values)

            thresh = 100/min(50, k + 1)
            # p=4
            # weights = weights * np.exp(((avg - values) / thresh*p))
            #weights = np.where(values == np.max(values), weights * np.exp(np.sign((avg - values)) / thresh * p * (1 - u)),
            #weights * np.exp((avg - values) / thresh * (1 - u)))

            # weights = weights * np.exp((avg - values) / thresh)

            gradient = u_history[-1] - u_history[-2]
            p = np.where(values == np.max(values), p + 1, 1)

            # weights = 2 * weights / (1 + np.exp(-(avg - values)))

            # weights = weights + ((avg - values)/ np.linalg.norm(avg - values))*thresh

            holo = self.holo_weights_and_phases(weights, solution)

            self.to_slm(holo)

            shot = self.take_shot()

            values = self.check_intensities(shot)
            u = self.uniformity(values)
            u_history.append(u)
            if np.max(u_history) > best:
                best_sol = weights

            print('Iteration                ::  ', k)
            print('BackLoop  [Intensities]  ::  ', values)
            print('BackLoop  [Weights]      ::  ', weights)
            print('BackLoop  [Uniformity]   ::  ', u)
            print()

            # plt.plot([i for i in range(len(u_history))], u_history)
            # plt.title(f'{k}')
            axs[0].clear()
            axs[0].plot([i for i in range(len(u_history))], u_history)

            axs[1].clear()
            axs[1].bar([i for i in range(len(values))], values)

            axs[2].clear()
            axs[2].bar([i for i in range(self.num_traps)], weights)

            axs[3].clear()
            axs[3].bar([i for i in range(self.num_traps)], p)

            axs[0].set_ylabel('Uniformity')
            axs[1].set_ylabel('Intensities')
            axs[2].set_ylabel('Weights')
            axs[3].set_ylabel('P')

            plt.title(f'{k},   u = {u}')

            plt.draw()

            plt.gcf().canvas.flush_events()

        print('BackLoop  [Best]             ::  ', best_sol)


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
