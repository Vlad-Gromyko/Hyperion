from customtkinter import CTkFrame

from core.services.abstract_plugin import AbstractPlugin
from core.event_bus import Event, EventBus

import customtkinter as ctk
import tkinter
import tksheet
from core.widgets.mask import MaskView

import numpy as np
import matplotlib.pyplot as plt
import pygad
import cv2


class Plugin(AbstractPlugin):
    def __init__(self, master, sub_master, *args, **kwargs):
        super().__init__(master, sub_master, *args, name='Коррекция Пучка\nЦернике', width=500, height=40, **kwargs)

        self.top_frame.grid_forget()

        self.notebook = ctk.CTkTabview(self.frame, segmented_button_selected_color='#000',
                                       text_color='#1E90FF', segmented_button_selected_hover_color='#00008B')
        self.notebook.grid()
        self.notebook.add('Выборка')

        self.sheet = tksheet.Sheet(self.notebook.tab('Выборка'), theme='dark blue', width=520,
                                   headers=[str(i) for i in range(4, 21)])

        self.sheet.grid(padx=5, pady=5, sticky='nsew', columnspan=2)

        self.sheet.enable_bindings()

        self.sheet.disable_bindings('copy', 'rc_insert_column', 'paste', 'cut', 'Delete', 'Edit cell', 'Delete columns')

        self.sheet.set_options(insert_row_label='Добавить Хромосому')
        self.sheet.set_options(delete_rows_label='Удалить Хромосому')
        self.sheet.set_options(insert_rows_above_label='Добавить Хромосому \u25B2')
        self.sheet.set_options(insert_rows_below_label='Добавить Хромосому \u25BC')
        self.sheet.set_options(delete_rows_label='Удалить Хромосому')

        frame = ctk.CTkFrame(self.notebook.tab('Выборка'))
        frame.grid(row=1, column=0)

        ctk.CTkButton(frame, text='Добавить случайные\nхромосомы Цернике', command=self.add_masks, fg_color='#000',
                      text_color='#1E90FF', ).grid(row=1, columnspan=2)

        ctk.CTkLabel(frame, text='Количество', bg_color='#000').grid(row=2, column=0, pady=5, sticky='nsew')
        self.num = ctk.CTkEntry(frame, width=50, bg_color='#1E90FF')
        self.num.insert(0, '100')
        self.num.grid(row=2, column=1, pady=5)

        self.notebook.add('Детекция')

        self.back_holo = None
        self.shift_holo = None

        ctk.CTkLabel(self.notebook.tab('Детекция'), text='Фон').grid(row=0, column=0)
        ctk.CTkLabel(self.notebook.tab('Детекция'), text='Смещение').grid(row=0, column=1)

        frame2 = ctk.CTkFrame(self.notebook.tab('Детекция'))
        frame2.grid(row=3)

        ctk.CTkLabel(frame2, text='Метод поиска центра:', fg_color='#000').grid(row=0)
        self.radio_var = tkinter.IntVar(value=0)
        radiobutton_1 = ctk.CTkRadioButton(frame2, text="Автоматически", variable=self.radio_var, value=0)
        radiobutton_2 = ctk.CTkRadioButton(frame2, text="Центр ROI", variable=self.radio_var, value=1)

        radiobutton_1.grid(row=1)
        radiobutton_2.grid(row=2)

        radiobutton_1.select()

        ctk.CTkButton(self.frame, text='Запустить генетический алгоритм', fg_color='#000',
                      text_color='#1E90FF', command=self.start_algo).grid(row=1, column=0, sticky='nsew', padx=5,
                                                                          pady=5)

        self.notebook.add('Параметры')

        frame_num_iterations = ctk.CTkFrame(self.notebook.tab('Параметры'))
        frame_num_iterations.grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(frame_num_iterations, text='Число поколений:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.num_iterations = ctk.CTkEntry(frame_num_iterations, width=45, bg_color='#1E90FF')
        self.num_iterations.insert(0, '100')
        self.num_iterations.grid(row=0, column=1)

        frame_crossover = ctk.CTkFrame(self.notebook.tab('Параметры'))
        frame_crossover.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(frame_crossover, text='Тип кроссинговера:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)

        self.combobox_cross = ctk.StringVar(value="single_point")
        self.combobox_cross = ctk.CTkComboBox(frame_crossover,
                                              values=["single_point", "two_points", 'uniform', 'scattered'],
                                              variable=self.combobox_cross)
        self.combobox_cross.set("single_point")
        self.combobox_cross.grid(row=0, column=1)

        frame_mutations = ctk.CTkFrame(self.notebook.tab('Параметры'))
        frame_mutations.grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkLabel(frame_mutations, text='Вероятность мутации %:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.mutations = ctk.CTkEntry(frame_mutations, width=45, bg_color='#1E90FF')
        self.mutations.insert(0, '10')
        self.mutations.grid(row=0, column=1)

        frame_par_mating = ctk.CTkFrame(self.notebook.tab('Параметры'))
        frame_par_mating.grid(row=2, column=0, padx=5, pady=5)
        ctk.CTkLabel(frame_par_mating, text='Кол-во родительских\nхромосом :', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.par_mating = ctk.CTkEntry(frame_par_mating, width=45, bg_color='#1E90FF')
        self.par_mating.insert(0, '10')
        self.par_mating.grid(row=0, column=1)

        frame_sol_pop = ctk.CTkFrame(self.notebook.tab('Параметры'))
        frame_sol_pop.grid(row=3, column=0, padx=5, pady=5)
        ctk.CTkLabel(frame_sol_pop, text='Число решений :', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.sol_pop = ctk.CTkEntry(frame_sol_pop, width=45, bg_color='#1E90FF')
        self.sol_pop.insert(0, '8')
        self.sol_pop.grid(row=0, column=1)

        frame_elitism = ctk.CTkFrame(self.notebook.tab('Параметры'))
        frame_elitism.grid(row=4, column=0, padx=5, pady=5)
        ctk.CTkLabel(frame_elitism, text='Элитизм :', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.elitism = ctk.CTkEntry(frame_elitism, width=45, bg_color='#1E90FF')
        self.elitism.insert(0, '3')
        self.elitism.grid(row=0, column=1)

        self.history = []
        self.counter = 0
        plt.ion()

        self.center_x = 0
        self.center_y = 0

        self.background = None

    def start_algo(self):
        self.history = []
        self.event_bus.raise_event(Event('TURN_ON_SLM'))
        self.event_bus.raise_event(Event('TO_SLM', self.back_holo.get_array()))
        cv2.waitKey(1)

        self.background = self.event_bus.raise_request(Event('TAKE_SHOT'))

        self.event_bus.raise_event(Event('TO_SLM', self.shift_holo.get_array()))
        cv2.waitKey(1)

        shifted = self.event_bus.raise_request(Event('TAKE_SHOT'))
        show = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
        b, g, r = cv2.split(show)

        if self.radio_var.get() == 0:
            self.center_x, self.center_y = center_place(np.abs(self.background - shifted))
        else:
            x_l, y_l, x_r, y_r = self.event_bus.raise_request(Event('TAKE_ROI'))
            self.center_x = (x_l + x_r) / 2
            self.center_y = (y_l + y_r) / 2

        b[int(self.center_y)] = 0
        b[:, int(self.center_x)] = 0

        g[int(self.center_y)] = 255
        g[:, int(self.center_x)] = 255

        r[int(self.center_y)] = 0
        r[:, int(self.center_x)] = 0

        cv2.imshow('CONTOUR', cv2.merge([b, g, r]))

        print('Координаты Центра Пятна:', self.center_x, self.center_y)

        def fitness_func(ga_instance, solution, solution_idx):
            test = self.event_bus.raise_result(Event('COMPUTE_ZERNIKE_BY_SPEC', solution)) + self.shift_holo.get_array()
            test = test % (2 * np.pi)
            self.event_bus.raise_event(Event('TO_SLM', test))
            cv2.waitKey(1)

            shot = self.event_bus.raise_request(Event('TAKE_SHOT'))

            candidate = np.abs(shot - self.background)
            ret, candidate = cv2.threshold(candidate, 50, 255, cv2.THRESH_TOZERO)

            center, contour, contours = draw_centered_contour(candidate)
            if contour is None:
                return -10000000
            shot_x, shot_y = center

            fitness_distance = - euclid_distance(self.center_x, self.center_y, shot_x, shot_y)

            compactness = (4 * np.pi * cv2.arcLength(contour, True) ** 2 + 0.0000001) / (
                        cv2.contourArea(contour) + 0.0000001)
            fitness_compactness = abs(1 / compactness)

            fitness_count_contours = - len(contours)

            return fitness_distance + fitness_compactness + fitness_count_contours

        def on_fitness(ga_instance, population_fitness):
            best = max(population_fitness)
            self.history.append(best)

            self.event_bus.raise_result(Event('PROGRESS_UPDATE',
                                              {'value': self.counter + 1,
                                               'max_value': int(self.num_iterations.get()),
                                               'name': 'Поколений Генетического Алгоритма'}))

            self.counter += 1

            plt.clf()
            plt.plot([i for i in range(len(self.history))], self.history)
            plt.draw()
            plt.gcf().canvas.flush_events()

        fitness_function = fitness_func

        num_generations = int(self.num_iterations.get())
        num_parents_mating = int(self.par_mating.get())

        sol_per_pop = int(self.sol_pop.get())

        crossover_type = self.combobox_cross.get()

        mutation_type = "random"
        mutation_percent_genes = 10

        keep_elitism = int(self.elitism.get())

        population = self.prepare_population()

        num_genes = len(population[0])

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               keep_elitism=keep_elitism,
                               initial_population=population,
                               on_fitness=on_fitness)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        best = self.event_bus.raise_result(Event('COMPUTE_ZERNIKE_BY_SPEC', solution)) + self.shift_holo.get_array()
        self.event_bus.raise_event(Event('TO_SLM', best))


    def prepare_population(self):
        population = []
        for i in range(self.sheet.get_total_rows()):
            spec = self.sheet[i].data
            population.append(spec)
        return population

    def add_masks(self):
        for i in range(int(self.num.get())):
            self.sheet.insert_row(list(np.random.uniform(-2, 2, 17)))

    def start(self):
        res_x = self.event_bus.raise_request(Event('SLM_WIDTH'))
        res_y = self.event_bus.raise_request(Event('SLM_HEIGHT'))
        gray = self.event_bus.raise_request(Event('SLM_GRAY'))

        self.back_holo = MaskView(self.notebook.tab('Детекция'), np.zeros((res_y, res_x)), gray,
                                  small_res_x=240, small_res_y=150)
        self.back_holo.grid(row=1, column=0)
        self.back_holo.add_menu_command('Взять из Цернике', lambda: self.back_holo.set_array(
            self.event_bus.raise_request(Event('TAKE_ZERNIKE'))))

        self.shift_holo = MaskView(self.notebook.tab('Детекция'), np.zeros((res_y, res_x)), gray,
                                   small_res_x=240, small_res_y=150)
        self.shift_holo.grid(row=1, column=1)
        self.shift_holo.add_menu_command('Взять из Цернике',
                                         lambda: self.shift_holo.set_array(
                                             self.event_bus.raise_request(Event('TAKE_ZERNIKE'))))

    def __del__(self):
        plt.ioff()


def center_place(image):
    res_y, res_x = np.shape(image)
    ax = np.linspace(0, res_x, res_x)
    ay = np.linspace(0, res_y, res_y)

    ax, ay = np.meshgrid(ax, ay)

    x_c = int(np.sum(ax * image) / np.sum(image))
    y_c = int(np.sum(ay * image) / np.sum(image))

    return x_c, y_c


def center_the_contour_all(shot):
    contours, hierarchy = cv2.findContours(shot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_x, c_y = center_place(shot)

    item = None
    for contour in contours:
        distance = cv2.pointPolygonTest(contour, (c_x, c_y), False)
        if distance >= 0:
            item = contour
    return (c_x, c_y), item, contours


def draw_centered_contour(shot):
    center, contour, contours = center_the_contour_all(shot)
    shot = cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(shot, [contour], -1, (0, 255, 0), 1)
    b, g, r = cv2.split(shot)

    c_x, c_y = center
    b[int(c_y)] = 0
    b[:, int(c_x)] = 0

    g[int(c_y)] = 255
    g[:, int(c_x)] = 255

    r[int(c_y)] = 0
    r[:, int(c_x)] = 0

    cv2.imshow('CONTOUR', cv2.merge([b, g, r]))

    return center, contour, contours


def euclid_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
