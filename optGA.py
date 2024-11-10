import cv2

from profilehooks import profile

from optics import *
import pygad

from optics import CoolCamera


class GeneticAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine,
                 trap_vision: Union[TrapSimulator, TrapVision],
                 iterations: int):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

    @profile(filename='ga2x2camera.prof', stdout=False)
    def run(self):
        def fitness_func(ga_instance, solution, solution_idx):
            holo = self.trap_machine.phase_holo_traps(solution)

            # self.slm.translate(holo)
            self.trap_vision.to_slm(holo)
            shot = self.trap_vision.take_shot()

            # plt.imshow(shot, cmap='hot')
            # plt.title(f'{solution_idx}')
            # plt.draw()
            # plt.gcf().canvas.flush_events()

            # self.slm.translate(holo)

            values = self.trap_vision.check_intensities(shot)

            return self.uniformity(values) / 2 + np.sum(values) / 2 / self.trap_machine.num_traps

        fitness_function = fitness_func

        num_genes = self.trap_machine.num_traps

        def on_generation(ga_instance):
            print(f"Generation = {ga_instance.generations_completed}")
            print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

        ga_instance = pygad.GA(num_generations=self.iterations,
                               num_parents_mating=5,
                               fitness_func=fitness_function,
                               sol_per_pop=10,
                               num_genes=num_genes,
                               crossover_type='single_point',
                               mutation_type='random',
                               mutation_percent_genes=10,
                               on_generation=on_generation,
                               init_range_low=-4 * np.pi,
                               init_range_high=4 * np.pi,
                               random_mutation_min_val=-4 * np.pi,
                               random_mutation_max_val=4 * np.pi,
                               keep_parents=2)

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        return solution


if __name__ == '__main__':
    _slm = SLM()
    _camera = CoolCamera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (2, 2), _slm)

    sim = TrapSimulator(_camera, _tr, _slm, search_radius=15)
    sim.register()
    sim.show_registered()


    alg = GeneticAlgorithm(_slm, _camera, _tr, sim, 40)

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
