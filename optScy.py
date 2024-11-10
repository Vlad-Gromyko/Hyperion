import cv2
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

    @profile(filename='try2x2.prof', stdout=False)
    def run(self):
        def fitness(solution):
            holo = self.trap_machine.numba_holo_traps(solution)
            # self.slm.translate(holo)
            self.trap_vision.to_slm(holo)

            shot = self.trap_vision.take_shot()

            intensities = self.trap_vision.check_intensities(shot)
            return 1 - self.uniformity(intensities)

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            print('X          = ', intermediate_result.x)
            print('Uniformity = ', 1 - fitness(intermediate_result.x))

        solution = np.random.uniform(low=0.0, high=1.0, size=self.trap_machine.num_traps)
        solution = np.asarray(solution)

        result = scipy.optimize.minimize(fitness, solution, args=(),
                                         callback=callback, options={'maxiter': int(self.iterations), 'disp': True})

        return result.x


if __name__ == '__main__':
    _slm = SLM()
    _camera = CoolCamera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (5, 5), _slm)

    sim = TrapSimulator(_camera, _tr, _slm, search_radius=15)
    sim.register()
    # sim.show_registered()

    alg = TryAlgorithm(_slm, _camera, _tr, sim, 30, 1)

    sol = alg.run()
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
