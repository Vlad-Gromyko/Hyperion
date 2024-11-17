import time

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
    def run(self, iterations):
        np.random.seed(42)
        starter = np.random.uniform(0, 2 * np.pi, (self.slm.height, self.slm.width))

        first_sol = np.random.uniform(0, 2 * np.pi, self.num_traps)

        st_time = time.time()
        phase, solution = gs(self.x_traps, self.y_traps,
                             self.wave, self.focus, self.slm.x, self.slm.y, starter, first_sol, 30)
        print('GS Compute-Time          ::  ', time.time() - st_time)
        print()
        print()

        phase = phase + np.pi

        self.slm.translate(phase)

        shot = self.vision.take_shot()
        values = self.check_intensities(shot)
        u = self.uniformity(values)

        print('After GS  [Intensities]  ::  ', values)
        print('After GS  [Uniformity]   ::  ', u)
        print('After GS  [Phases]       ::  ', solution)

        print('Start BackLoop...')

        weights = np.ones(self.num_traps)
        for k in range(iterations):

            avg = np.average(values)

            weights = weights * np.exp(avg - values)

            holo = self.holo_weights_and_phases(weights, solution)

            self.slm.translate(holo)

            shot = self.vision.take_shot()

            values = self.check_intensities(shot)
            u = self.uniformity(values)

            print('Iteration                ::  ', k)
            print('BackLoop  [Intensities]  ::  ', values)
            print('BackLoop  [Weights]      ::  ', weights)
            print('BackLoop  [Uniformity]   ::  ', u)
            print()




if __name__ == '__main__':
    exp = GSplusWeight()

    exp.add_array(0, 0, 120 * UM, 120 * UM, 3, 3)
    #exp.add_circle_array(0,0, 120 * UM, 10)
    #exp.show_trap_map()

    exp.register_traps()


    exp.run(10)

