import cv2

from profilehooks import profile

from optics import *
import pygad

from optics import CoolCamera


class GeneticAlgorithm(Algorithm):
    def __init__(self, slm: SLM, camera: Camera, trap_machine: TrapMachine, trap_vision: Union[TrapSimulator, TrapVision],
                 iterations: int):
        super().__init__(slm, camera, trap_machine, trap_vision, iterations)

    @profile(filename='ga2x2camera.prof')
    def run(self):
        def fitness_func(ga_instance, solution, solution_idx):
            holo = self.trap_machine.holo_traps(solution)

            self.slm.translate(holo)
            self.trap_vision.to_slm(holo)
            shot = self.trap_vision.take_shot()

            #plt.imshow(shot, cmap='hot')
            #plt.title(f'{solution_idx}')
            #plt.draw()
            #plt.gcf().canvas.flush_events()

            # self.slm.translate(holo)

            values = self.trap_vision.check_intensities(shot)
            return self.uniformity(values)

        fitness_function = fitness_func

        num_genes = self.trap_machine.num_traps

        def on_generation(ga_instance):
            print(f"Generation = {ga_instance.generations_completed}")
            print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

        ga_instance = pygad.GA(num_generations=self.iterations,
                               num_parents_mating=10,
                               fitness_func=fitness_function,
                               sol_per_pop=20,
                               num_genes=num_genes,
                               crossover_type='single_point',
                               mutation_type='random',
                               mutation_percent_genes=20,
                               on_generation=on_generation,
                               init_range_low=0.1,
                               init_range_high=4)

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        return self.trap_machine.holo_traps(solution)


if __name__ == '__main__':
    _slm = SLM()
    _camera = CoolCamera()
    _tr = TrapMachine((0, 0), (120 * UM, 120 * UM), (2, 2), _slm)

    sim = TrapVision(_camera, _tr, _slm, search_radius=5)
    sim.register()
    # sim.show_registered()

    alg = GeneticAlgorithm(_slm, _camera, _tr, sim, 20)

    ph = alg.run()

    #i = sim.propagate(sim.holo_box(ph))
    #plt.ioff()
    #plt.title('Result')
    #plt.imshow(i, cmap='hot')
    # plt.show()
