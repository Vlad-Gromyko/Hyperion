from optics import *


class HotaAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine,
                 trap_vision: Union[TrapVision, TrapSimulator],
                 iterations: int,
                 starter: np.ndarray, user_weights: np.ndarray):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

        self.starter = starter
        self.phase = np.zeros_like(starter, dtype='float64')
        self.user_weights = np.asarray(user_weights)

        self.v_list = np.ones(self.trap_machine.num_traps, dtype='complex128')
        self.w_list = np.ones(self.trap_machine.num_traps, dtype='float64')
        self.w_list_before = None

        self.shots = []

    def run(self):

        plt.ion()
        anti_user_weights = 1 / self.user_weights

        # self.calc_v()

        # bar = Bar('Алгоритм', max=int(self.iterations))
        for k in range(int(self.iterations)):
            w_list_before = self.w_list
            avg = np.average(np.abs(self.v_list), weights=anti_user_weights)

            self.w_list = avg / np.abs(self.v_list) * self.user_weights * w_list_before

            summ = np.zeros_like(self.starter, dtype=np.complex128)
            for i in range(self.trap_machine.num_traps):
                trap = self.trap_machine.holo_trap(self.trap_machine.x_traps[i], self.trap_machine.y_traps[i])
                summ = summ + np.exp(1j * trap) * self.user_weights[i] * self.w_list[i] * self.v_list[i] / np.abs(
                    self.v_list[i])

            self.phase = np.angle(summ) + np.pi

            # self.slm.translate(self.phase)

            self.trap_vision.to_slm(self.phase)
            shot = self.trap_vision.take_shot()

            self.shots.append(shot)

            intensities = np.asarray(self.trap_vision.check_intensities(shot))
            print(intensities)
            self.v_list = intensities + 10
            self.v_list = self.v_list / np.max(np.abs(self.v_list))

            print('intensities ', intensities)

            self.history['uniformity_history'].append(self.uniformity(intensities))

            cv2.waitKey(1)

            plt.clf()

            # plt.plot([i for i in range(len(self.history['uniformity_history']))], self.history['uniformity_history'])

            im = plt.imshow(shot, cmap='hot')
            plt.title(f'{k + 1}')
            plt.draw()
            plt.gcf().canvas.flush_events()
            # bar.next()
            print('HOTA ', k + 1)
        # bar.finish()

        return self.phase

    def calc_v(self):
        area = self.slm.mesh.width * self.slm.mesh.height
        for i in range(self.trap_machine.num_traps):
            trap = self.trap_machine.holo_trap(self.trap_machine.x_traps[i], self.trap_machine.y_traps[i])
            self.v_list[i] = 1 / area * np.sum(np.exp(self.phase - trap))

        return self.v_list

    def measure_v(self, shot):
        values = self.trap_vision.check_intensities(shot)
        self.v_list = values
        return self.v_list


if __name__ == '__main__':
    _slm = SLM()
    _camera = Camera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (3, 3), _slm)

    sim = TrapSimulator(_camera, _tr, _slm, search_radius=5)
    sim.register()
    # sim.show_registered()

    user = np.asarray([1 for i in range(_tr.num_traps)])
    alg = HotaAlgorithm(_slm, _camera, _tr, sim, 40, np.zeros((_slm.mesh.height, _slm.mesh.width)), user)
    ph = alg.run()

    i = sim.propagate(sim.holo_box(ph))
    plt.ioff()
    plt.title('Result')
    plt.imshow(i, cmap='hot')
    plt.show()
    cv2.waitKey(0)