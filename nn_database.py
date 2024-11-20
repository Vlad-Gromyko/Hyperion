import cv2
import numpy as np


from check_traps import uniformity, intensity
from optics import *
import pygad


import copy


class TryAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine,
                 trap_vision: Union[TrapSimulator, TrapVision],
                 iterations: int, step: float):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

        self.step = step

    def write_file(self, solution, u, values, k):
        soluttion = [str(solution[i]) for i in range(len(solution))]
        values = [str(values[i]) for i in range(len(values))]
        u = str(u)
        with open(rf'3x3_x=1000_dx=120\{k}.txt', 'w') as f:
            f.write(' '.join(soluttion)+'\n')
            f.write(' '.join(values) + '\n')
            f.write(u)



    def run(self):
        plt.ion()
        for k in range(int(self.iterations)):
            solution = np.random.uniform(-10, 10, self.trap_machine.num_traps)
            intensities, u, shot = self.check(solution + 1)
            intensities = np.asarray(intensities, dtype='float64')

            self.write_file(solution, u, intensities, k)

            plt.clf()

            plt.draw()
            plt.imshow(shot, cmap='hot')

            plt.gcf().canvas.flush_events()

            print(f'Generation    = {k}')
            print(f'Solution      = {solution}')

            print(f'Intensities   = {intensities}')
            print(f'Uniformity    = {u}')
            print()

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
        intensities = np.asarray(intensities, dtype='float64')

        return intensities, self.uniformity(intensities), shot


if __name__ == '__main__':
    _slm = SLM()
    _camera = Camera()
    _tr = TrapMachine((1000 * UM, 0), (120 * UM, 120 * UM), (3, 3), _slm)

    sim = TrapVision(_camera, _tr, _slm, search_radius=15)
    sim.register()
    # sim.show_registered()

    alg = TryAlgorithm(_slm, _camera, _tr, sim, 10000, 200)

    sol = alg.run()
    print(sol)
    #result = sim.propagate(sim.holo_box(_tr.numba_holo_traps(sol)))

   # im = plt.imshow(result, cmap='nipy_spectral')
    #plt.colorbar(im)
    #plt.show()

    # i = sim.propagate(sim.holo_box(ph))
    # plt.ioff()
    # plt.title('Result')
    # plt.imshow(i, cmap='hot')
    # plt.show()
