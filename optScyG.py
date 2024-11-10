import cv2
import matplotlib.pyplot as plt
import numpy as np

from profilehooks import profile

from check_traps import uniformity
from optics import *
import pygad

import scipy


class TryAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine,
                 trap_vision: Union[TrapSimulator, TrapVision],
                 iterations: int, step: float):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

        self.step = step
        self.k = 0
        self.history = []

    @profile(filename='try2x2.prof', stdout=False)
    def run(self):
        def fitness(solution):
            print(solution)
            holo = self.trap_machine.numba_true(solution)
            # self.slm.translate(holo)
            self.trap_vision.to_slm(holo)

            shot = self.trap_vision.take_shot()

            intensities = self.trap_vision.check_intensities(shot)
            u = np.sum(intensities) / self.trap_machine.num_traps
            u = u + self.uniformity(intensities)
            u = u - np.std(intensities)
            u = u/2 if np.count_nonzero(intensities) else 0

            self.history.append(u)

            plt.plot([i for i in range(len(self.history))], self.history)
            plt.title(f'{self.k}')
            plt.draw()
            plt.gcf().canvas.flush_events()
            self.k += 1


            return 1 - u

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            print('X          = ', intermediate_result.x)
            print('Fitness    = ', 1 - fitness(intermediate_result.x))


        bounds = [(1, 2) for i in range(int(self.trap_machine.num_traps))]
        solution = scipy.optimize.differential_evolution(fitness, bounds,
                                                       maxiter=20, popsize=1, callback=callback)

        return solution.x


if __name__ == '__main__':
    _slm = SLM()
    _camera = CoolCamera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (1, 3), _slm)

    sim = TrapSimulator(_camera, _tr, _slm, search_radius=15)
    sim.register()
    # sim.show_registered()

    alg = TryAlgorithm(_slm, _camera, _tr, sim, 20, 1)
    plt.ion()
    sol = alg.run()

    plt.ioff()
    print(sol)
    result = sim.propagate(sim.holo_box(_tr.numba_holo_traps(sol)))

    im = plt.imshow(result, cmap='nipy_spectral')
    plt.colorbar(im)
    plt.show()

    # i = sim.propagate(sim.holo_box(ph))
    # plt.ioff()
    # plt.title('Result')
    # plt.imshow(i, cmap='hot')
    # plt.show()
