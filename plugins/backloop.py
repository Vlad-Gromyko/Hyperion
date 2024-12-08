from PIL.Image import Image
from PIL.ImagePalette import wedge

from core.services.abstract_plugin import AbstractPlugin
from core.event_bus import Event, EventBus

import customtkinter as ctk

from core.widgets.mask import MaskView

import numpy as np

import cv2

from PIL import Image


class Plugin(AbstractPlugin):
    def __init__(self, master, sub_master, *args, **kwargs):
        super().__init__(master, sub_master, *args, name='Обратная связь', width=500, height=40, **kwargs)

        self.top_frame.grid_forget()

        self.frame_top_buttons = ctk.CTkFrame(self.frame, width=200, height=200)
        self.frame_top_buttons.grid(row=0, column=0, padx=5, pady=5)

        ctk.CTkButton(self.frame_top_buttons, text='\u2B9C', width=30, height=30,
                      command=lambda: self.add_move(dx=-1)).grid(row=1, column=0)
        ctk.CTkButton(self.frame_top_buttons, text='\u2B9D', width=30, height=30,
                      command=lambda: self.add_move(dy=-1)).grid(row=0, column=1)
        ctk.CTkButton(self.frame_top_buttons, text='\u2B9E', width=30, height=30,
                      command=lambda: self.add_move(dx=1)).grid(row=1, column=2)
        ctk.CTkButton(self.frame_top_buttons, text='\u2B9F', width=30, height=30,
                      command=lambda: self.add_move(dy=1)).grid(row=2, column=1)

        self.frame_entries = ctk.CTkFrame(self.frame)
        self.frame_entries.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(self.frame_entries, text='X (мкм) ').grid(row=0, column=0)
        ctk.CTkLabel(self.frame_entries, text='Y (мкм) ').grid(row=1, column=0)

        ctk.CTkLabel(self.frame_entries, text='  dX (мкм) ').grid(row=0, column=2)
        ctk.CTkLabel(self.frame_entries, text='  dY (мкм) ').grid(row=1, column=2)

        self.c_x = ctk.CTkEntry(self.frame_entries)
        self.c_x.grid(row=0, column=1)
        self.c_x.insert(0, '0')

        self.c_y = ctk.CTkEntry(self.frame_entries)
        self.c_y.grid(row=1, column=1)
        self.c_y.insert(0, '0')

        self.d_x = ctk.CTkEntry(self.frame_entries)
        self.d_x.grid(row=0, column=3)
        self.d_x.insert(0, '200')

        self.d_y = ctk.CTkEntry(self.frame_entries)
        self.d_y.grid(row=1, column=3)
        self.d_y.insert(0, '200')

        self.screen = ctk.CTkLabel(self.frame, text=' ')
        self.screen.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.frame_buttons = ctk.CTkFrame(self.frame)
        self.frame_buttons.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        ctk.CTkButton(self.frame_buttons, text='Регистрация ловушек', command=self.register_traps).grid(row=0, column=0,
                                                                                                        padx=5, pady=5)
        self.button_start = ctk.CTkButton(self.frame_buttons, text='Начать оптимизацию', command=self.run_optimization)
        self.button_start.grid(row=1, column=0, padx=5, pady=5)

        self.button_pause = ctk.CTkButton(self.frame_buttons, text='Остановить оптимизацию', command=self.pause)
        self.button_pause.grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(self.frame_buttons, text='Прервать оптимизацию').grid(row=1, column=2, padx=5, pady=5)

        self.check_var = ctk.StringVar(value="off")
        self.checkbox = ctk.CTkSwitch(self.frame_buttons, text="Виртуальная камера", progress_color='#DC143C',
                                      text_color='#DC143C',
                                      variable=self.check_var, onvalue="on", offvalue="off")
        self.checkbox.grid(row=0, column=1, padx=5, pady=5)

        f = ctk.CTkFrame(self.frame_buttons)
        f.grid(row=0, column=2, padx=5, pady=5)

        ctk.CTkLabel(f, text='Размер окна : ').grid(row=0, column=0)

        self.search_radius = ctk.CTkEntry(f)
        self.search_radius.insert(0, '10')
        self.search_radius.grid(row=0, column=1)

        self.frame_results = ctk.CTkFrame(self.frame)
        self.frame_results.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        ctk.CTkLabel(self.frame_results, text='Лучшая маска').grid()
        self.best_holo = None

        self.best_uniformity = ctk.CTkLabel(self.frame_results, text='Лучшая однородность:')

        self.mesh_x = None
        self.mesh_y = None

        self.registered_x = []
        self.registered_y = []

        self.running = True


        self.design = []
        self.weights = []

        self.u_history = []
        self.weights_history = []
        self.intensities_history = []
        self.gradient_history = []

        self.best_weights = []
        self.best_uniformity = []
        self.best_intensities = []



    def run_optimization(self):
        if len(self.u_history) ==0:
            self.weights = np.random.uniform(1, 1.1, len(self.registered_y))
            self.event_bus.raise_event(Event('NEW_WEIGHTS', self.weights))

        if self.running:
            self.iteration()
            self.after(100, self.run_optimization)
        else:
            self.running = True

    def iteration(self):
        self.event_bus.raise_event(Event('HOTA'))
        holo = self.event_bus.raise_request(Event('TAKE_HOTA'))

        self.event_bus.raise_event(Event('TO_SLM', holo))

        if self.check_var.get() == 'on':
            self.event_bus.raise_event(Event('PROPAGATE'))
            shot = self.event_bus.raise_request(Event('PROPAGATED'))
        else:
            shot = self.event_bus.raise_request(Event("TAKE_SHOT"))

        values = self.check_intensities(shot)
        print(values)
        u = self.uniformity(values)
        avg = np.average(values)
        print(u)
        velocity =1
        thresh = min(len(self.u_history) + 0.1, 100)


        if (len(self.best_uniformity) == 0) or (u > self.best_uniformity[-1]):
            self.best_uniformity.append(u)
            self.best_weights.append(self.weights)

            print('best')

        self.weights_history.append(self.weights)
        self.intensities_history.append(values)
        self.u_history.append(u)


        self.weights = self.best_weights[-1] + velocity / thresh * (avg - values)
        self.event_bus.raise_event(Event('NEW_WEIGHTS', self.weights))


    @staticmethod
    def uniformity(values):
        return 1 - (np.max(values) - np.min(values)) / (np.max(values) + np.min(values))

    def check_intensities(self, shot):
        values = []
        for i in range(len(self.registered_y)):
            value = self.intensity(self.registered_x[i], self.registered_y[i], shot)
            self.draw_area(i, self.registered_y, self.registered_x, shot)
            values.append(value)

        return np.asarray(values) / np.max(values)

    def draw_area(self, i, x_list, y_list, shot):

        h, w = np.shape(shot)
        shot = shot / np.max(shot) * 255
        shot = np.asarray(shot, dtype='uint8')
        shot = cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR)

        search_radius = int(self.search_radius.get())
        for k in range(len(x_list)):
            shot = cv2.rectangle(shot, (x_list[k] - search_radius, y_list[k] - search_radius),
                                 (x_list[k] + search_radius, y_list[k] + search_radius), (255, 0, 0), 1)

        show = cv2.rectangle(shot, (x_list[i] - search_radius, y_list[i] - search_radius),
                             (x_list[i] + search_radius, y_list[i] + search_radius), (0, 255, 0), 1)

        image = ctk.CTkImage(light_image=Image.fromarray(show), size=(w // 5, h // 5))

        self.screen.configure(image=image)
        self.screen.update_idletasks()
        self.update_idletasks()

    def pause(self):

        self.running = False

    def start(self):
        shot = self.event_bus.raise_request(Event('TAKE_SHOT'))
        y, x = np.shape(shot)
        shot = ctk.CTkImage(light_image=Image.fromarray(shot), size=(x // 4, y // 4))
        self.screen.configure(image=shot)

        res_x = self.event_bus.raise_request(Event('SLM_WIDTH'))
        res_y = self.event_bus.raise_request(Event('SLM_HEIGHT'))
        gray = self.event_bus.raise_request(Event('SLM_GRAY'))
        pixel = self.event_bus.raise_request(Event('SLM_PITCH'))

        _x = np.linspace(- res_x // 2 * pixel, res_x // 2 * pixel, res_x)
        _y = np.linspace(-res_y // 2 * pixel, res_y // 2 * pixel, res_y)

        x, y = np.meshgrid(_x, _y)

        self.mesh_x = x
        self.mesh_y = y

        self.best_holo = MaskView(master=self.frame_results, array=np.zeros((res_y, res_x)), slm_gray_edge=gray,
                                  small_res_x=160,
                                  small_res_y=100)

        self.best_holo.grid(row=0, column=0, padx=5, pady=5)

        self.best_holo.add_menu_command('Отправить на SLM',
                                        lambda: self.event_bus.raise_event(Event('TO_SLM', self.best_holo.get_array())))

        self.best_holo.add_menu_command('Отправить в Атлас',
                                        lambda: self.event_bus.raise_event(
                                            Event('TO_ATLAS', self.best_holo.get_array())))
        self.best_holo.add_menu_command('Отправить в Аккумулятор',
                                        lambda: self.event_bus.raise_event(
                                            Event('TO_ACCUMULATOR', self.best_holo.get_array())))



    def register_traps(self):
        self.registered_x = []
        self.registered_y = []

        # self.back = self.event_bus.raise_request(Event('TAKE_SHOT'))

        # if self.check_var.get() == 'on':
        # self.event_bus.raise_event(Event('PROPAGATE'))
        # self.back = self.event_bus.raise_request(Event('PROPAGATED'))
        # else:
        #   self.back = self.event_bus.raise_request(Event('TAKE_SHOT'))

        traps_x, traps_y, weights = self.event_bus.raise_request(Event('TRAPS_SPECS'))
        self.design = weights
        num_traps = len(traps_x)
        wave = self.event_bus.raise_request(Event('WAVE'))
        focus = self.event_bus.raise_request(Event('FOCUS'))

        for i in range(num_traps):
            x_trap = traps_x[i]
            y_trap = traps_y[i]
            holo = 2 * np.pi / wave / focus * (self.mesh_x * x_trap + self.mesh_y * y_trap) % (2 * np.pi)

            self.event_bus.raise_event(Event('TO_SLM', holo))
            cv2.waitKey(1)

            if self.check_var.get() == 'on':
                self.event_bus.raise_event(Event('PROPAGATE'))
                shot = self.event_bus.raise_request(Event('PROPAGATED'))
            else:
                shot = self.event_bus.raise_request(Event("TAKE_SHOT"))

            y, x = self.find_trap(np.abs(shot))
            self.registered_x.append(x)
            self.registered_y.append(y)
            self.event_bus.raise_event(
                Event('PROGRESS_UPDATE', data={'value': i + 1, 'max_value': num_traps, 'name': 'Регистрация ловушек'}))

            self.draw_registered(shot)

    def draw_registered(self, shot):
        h, w = np.shape(shot)
        shot = shot / np.max(shot) * 255
        shot = np.asarray(shot, dtype='uint8')
        shot = cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR)

        search_radius = int(self.search_radius.get())
        y_list = self.registered_x
        x_list = self.registered_y
        for k in range(len(self.registered_y)):
            shot = cv2.rectangle(shot, (x_list[k] - search_radius, y_list[k] - search_radius),
                                 (x_list[k] + search_radius, y_list[k] + search_radius), (255, 0, 0), 1)

        image = ctk.CTkImage(light_image=Image.fromarray(shot), size=(w // 5, h // 5))

        self.screen.configure(image=image)
        self.screen.update_idletasks()
        self.update_idletasks()

    @staticmethod
    def find_center(image):
        res_y, res_x = np.shape(image)
        ax = np.linspace(0, res_x, res_x)
        ay = np.linspace(0, res_y, res_y)

        ax, ay = np.meshgrid(ax, ay)

        x_c = int(np.sum(ax * image) / np.sum(image))

        y_c = int(np.sum(ay * image) / np.sum(image))
        return x_c, y_c

    def find_trap(self, array):
        spot = np.max(array)

        mask = np.where(array == spot, array, 0)
        x, y = self.find_center(mask)
        return x, y

    def masked(self, x, y, array):
        mask = np.zeros_like(array)

        # for iy in range(height):
        # for ix in range(width):
        # if (ix - x) ** 2 + (iy - y) ** 2 <= self.search_radius ** 2:
        # mask[iy, ix] = 1
        search_radius = int(self.search_radius.get())
        mask[x - search_radius: x + search_radius, y - search_radius: y + search_radius] = 1

        return mask

    def intensity(self, x, y, shot):
        mask = self.masked(x, y, shot)
        return np.max(shot * mask)

    def add_move(self, dx=0, dy=0):
        c_x = float(self.c_x.get())
        c_y = float(self.c_y.get())

        c_x += float(self.d_x.get()) * dx
        c_y += float(self.d_y.get()) * dy

        self.c_x.delete(0, 'end')
        self.c_y.delete(0, 'end')

        self.c_x.insert(0, str(c_x))
        self.c_y.insert(0, str(c_y))

        c_x *= 10 ** -6
        c_y *= 10 ** -6

        wave = self.event_bus.raise_request(Event('WAVE'))
        focus = self.event_bus.raise_request(Event('FOCUS'))

        holo = 2 * np.pi / wave / focus * (self.mesh_x * c_x + self.mesh_y * c_y) % (2 * np.pi)

        self.event_bus.raise_event(Event('TO_SLM', data=holo))

        if self.check_var.get() == 'on':
            self.event_bus.raise_event(Event('PROPAGATE'))
            shot = self.event_bus.raise_request(Event('PROPAGATED'))
        else:
            shot = self.event_bus.raise_request(Event("TAKE_SHOT"))

        y, x = np.shape(shot)
        shot = ctk.CTkImage(light_image=Image.fromarray(shot), size=(x // 4, y // 4))
        self.screen.configure(image=shot)
