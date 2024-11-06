import numpy as np

import screeninfo
import cv2

import pygad

import matplotlib.pyplot as plt

import lab
from lab import Lab


class HotaGeneticAlgorithm:
    def __init__(self, x_c, y_c, x_d, y_d, x_n, y_n, laboratory: Lab,
                 num_generations=10,
                 num_parent_mating=4,
                 sol_per_pop=8,
                 init_range_low=0,
                 init_range_high=5,
                 parent_selection='sss',
                 keep_parents=1,
                 crossover='single_point'):
        self.lab = laboratory

        self.x_c, self.y_c = x_c, y_c
        self.x_n, self.y_n = x_n, y_n
        self.x_d, self.y_d = x_d, y_d

        self.num_generations = num_generations
        self.num_parent_mating = num_parent_mating
        self.sol_per_pop = sol_per_pop
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.parent_selection = parent_selection
        self.keep_parents = keep_parents
        self.crossover = crossover

        self.traps_x, self.traps_y = self.lab.prepare_traps(x_c, y_c, x_d, y_d, x_n, y_n)

        self.reg_x, self.reg_y = self.lab.registration(self.traps_x, self.traps_y)

        self.num_genes = len(self.reg_x)

        self.ga = None
        self.history = []

    def uniformity(self, values):
        minimum = np.min(values)
        maximum = np.max(values)

        return 1 - (maximum - minimum) / (minimum + maximum)

    def run(self):

        def fitness(ga_instance, solution, solution_idx):
            holo = self.lab.holo_traps(self.traps_x, self.traps_y, solution)
            self.lab.to_slm(holo)
            cv2.waitKey(1)

            shot = self.lab.take_shot()

            intensities = []

            for i in range(len(self.reg_x)):
                intensities.append(self.lab.intensity(self.reg_x[i], self.reg_y[i], shot))

            print(intensities)
            result = self.uniformity(intensities)

            return result

        plt.ion()


        def on_fitness(ga_instance, population_fitness):
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            print(f"Parameters of the best solution : {solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print()

            self.history.append(solution_fitness)
            plt.clf()

            plt.plot([i for i in range(len(self.history))], self.history)

            plt.draw()
            plt.gcf().canvas.flush_events()

        self.ga = pygad.GA(num_generations=4,
                           num_parents_mating=self.num_parent_mating,
                           fitness_func=fitness,
                           sol_per_pop=self.sol_per_pop,
                           num_genes=self.num_genes,
                           init_range_low=self.init_range_low,
                           init_range_high=self.init_range_high,
                           parent_selection_type=self.parent_selection,
                           keep_parents=self.keep_parents,
                           crossover_type=self.crossover,
                           on_fitness=on_fitness)

        plt.ioff()

        solution, solution_fitness, solution_idx = self.ga.best_solution(self.ga.last_generation_fitness)
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")


if __name__ == '__main__':
    laboratory = Lab()
    ga = HotaGeneticAlgorithm(0, 0, 200 * lab.UM, 200 * lab.UM, 2, 2, laboratory)

    ga.run()
