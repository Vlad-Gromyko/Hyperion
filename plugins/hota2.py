from core.services.abstract_plugin import AbstractPlugin
from core.event_bus import Event, EventBus

import customtkinter as ctk
from core.widgets.mask import MaskView

import numpy as np
import numba
import time


class Plugin(AbstractPlugin):
    def __init__(self, master, sub_master, *args, **kwargs):
        super().__init__(master, sub_master, *args, name='HOTA numba2', width=500, height=400, **kwargs)

        self.notebook = ctk.CTkTabview(self.frame, segmented_button_selected_color='#000',
                                              text_color='#7FFF00', segmented_button_selected_hover_color='#006400',)
        self.notebook.grid()

        self.notebook.add('Голограмма')
        self.notebook.add('Стартер')

        self.holo = None
        self.starter = None

        ctk.CTkButton(self.frame, text='Старт', command=self.algo_start,  fg_color='#000', text_color='#1E90FF',).grid()

        frame_iterations = ctk.CTkFrame(self.frame)
        frame_iterations.grid()
        ctk.CTkLabel(frame_iterations, text='Число Итераций:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.iterations = ctk.CTkEntry(frame_iterations, width=45, bg_color='#1E90FF')
        self.iterations.insert(0, '30')
        self.iterations.grid(row=0, column=1)

        self.timer = ctk.CTkLabel(self.frame, text='ВРЕМЯ : ', fg_color='#000')
        self.timer.grid()

    def algo_start(self):
        start = time.time()
        traps, weights = self.event_bus.raise_request(Event('TRAPS_SPECS'))
        iterations = int(self.iterations.get())
        starter = self.starter.get_array()

        result = mega_HOTA(traps, user_weights=weights, initial_phase=starter,
                           iterations=iterations) + np.pi

        self.holo.set_array(result)

        self.timer.configure(text=f'Время : {time.time() - start}')



    def start(self):
        res_x = self.event_bus.raise_request(Event('SLM_WIDTH'))
        res_y = self.event_bus.raise_request(Event('SLM_HEIGHT'))
        gray = self.event_bus.raise_request(Event('SLM_GRAY'))

        self.holo = MaskView(master=self.notebook.tab('Голограмма'), array=np.zeros((res_y, res_x)), slm_gray_edge=gray, small_res_x=320,
                             small_res_y=200)

        self.holo.grid(row=0, column=0, padx=5, pady=5)

        self.holo.add_menu_command('Отправить на SLM',
                                   lambda: self.event_bus.raise_event(Event('TO_SLM', self.holo.get_array())))

        self.holo.add_menu_command('Отправить в Атлас',
                                   lambda: self.event_bus.raise_event(Event('TO_ATLAS', self.holo.get_array())))
        self.holo.add_menu_command('Отправить в Аккумулятор',
                                   lambda: self.event_bus.raise_event(Event('TO_ACCUMULATOR', self.holo.get_array())))

        self.starter = MaskView(master=self.notebook.tab('Стартер'), array=np.zeros((res_y, res_x)), slm_gray_edge=gray, small_res_x=320,
                                small_res_y=200)

        self.starter.grid(row=0, column=0, padx=5, pady=5)

        self.starter.add_menu_command('Отправить на SLM',
                                      lambda: self.event_bus.raise_event(Event('TO_SLM', self.starter.get_array())))

        self.starter.add_menu_command('Отправить в Атлас',
                                      lambda: self.event_bus.raise_event(Event('TO_ATLAS', self.starter.get_array())))
        self.starter.add_menu_command('Отправить в Аккумулятор',
                                      lambda: self.event_bus.raise_event(
                                          Event('TO_ACCUMULATOR', self.starter.get_array())))

@numba.njit(fastmath=True, parallel=True)
def mega_HOTA(traps, user_weights, initial_phase, iterations):
    num_traps = len(user_weights)
    v_list = np.zeros_like(user_weights, dtype=np.complex128)
    area = np.shape(initial_phase)[0] * np.shape(initial_phase)[1]
    phase = np.zeros_like(initial_phase)

    w_list = np.ones(num_traps)


    for i in numba.prange(num_traps):
        v_list[i] = 1 / area * np.sum(np.exp(1j * (initial_phase - traps[i])))

    anti_user_weights = 1 / user_weights

    for k in range(iterations):
        w_list_before = w_list
        avg = np.average(np.abs(v_list), weights=anti_user_weights)

        w_list = avg / np.abs(v_list) * user_weights * w_list_before

        summ = np.zeros_like(initial_phase, dtype=np.complex128)
        for ip in range(num_traps):
            summ = summ + np.exp(1j * traps[ip]) * user_weights[ip] * v_list[ip] * w_list[ip] / np.abs(
                v_list[ip])
        phase = np.angle(summ)

        for iv in range(num_traps):
            v_list[iv] = 1 / area * np.sum(np.exp(1j * (phase - traps[iv])))

    return phase