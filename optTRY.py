import cv2
import numpy as np

from profilehooks import profile

from check_traps import uniformity, intensity
from optics import *
import pygad

from optics import CoolCamera

import copy


class TryAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine,
                 trap_vision: Union[TrapSimulator, TrapVision],
                 iterations: int, step: float):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

        self.step = step

    @profile(filename='try2x2.prof', stdout=False)
    def run(self):
        solution = np.ones(self.trap_machine.num_traps)
        solution = np.asarray(solution)
        plt.ion()
        for k in range(int(self.iterations)):
            intensities, u, shot = self.check(solution + 1)

            plt.clf()

            plt.imshow(shot, cmap='hot')

            plt.draw()
            plt.gcf().canvas.flush_events()

            print(f'Generation    = {k}')
            print(f'Solution      = {solution}')

            print(f'Intensities   = {intensities}')
            print(f'Uniformity    = {u}')
            print()

            coeff = 1 / np.min([self.step, k + 1])
            steps = (np.average(intensities) - intensities)
            solution = solution - steps / np.max(steps) * coeff
            # for i in range(self.trap_machine.num_traps):
        # if intensities[i] == np.min(intensities):
        # solution[i] += coeff * steps[i] / np.max(steps)

        return solution

    def check(self, values):
        holo = self.trap_machine.phase_holo_traps(values)
        # self.slm.translate(holo)
        self.trap_vision.to_slm(holo)

        shot = self.trap_vision.take_shot()

        intensities = self.trap_vision.check_intensities(shot)

        return intensities, self.uniformity(intensities), shot


if __name__ == '__main__':
    _slm = SLM()
    _camera = CoolCamera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (3, 3), _slm)

    sim = TrapSimulator(_camera, _tr, _slm, search_radius=15)
    sim.register()
    # sim.show_registered()

    alg = TryAlgorithm(_slm, _camera, _tr, sim, 1000, 200)

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
