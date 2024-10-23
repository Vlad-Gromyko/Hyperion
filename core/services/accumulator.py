import numpy as np

from core.services.service import Service
from core.event_bus import Event, EventBus

from core.widgets.mask import MaskView

import customtkinter as ctk


class Accumulator(Service):
    def __init__(self, master, width, height, gray, *args, **kwargs):
        super().__init__(master, *args, name='Атлас', width=250, height=1, **kwargs)

        self.mask = MaskView(self.top_frame, np.zeros((height, width)), gray, small_res_x=160, small_res_y=100)
        self.mask.grid()

        self.mask.add_menu_command('Отправить на SLM',
                                   lambda: self.event_bus.raise_event(Event('TO_SLM', self.mask.get_array())))

        self.mask.add_menu_command('Отправить в Атлас',
                                   lambda: self.event_bus.raise_event(Event('TO_ATLAS', self.mask.get_array())))
        self.mask.add_menu_command('Отправить в Аккумулятор',
                                   lambda: self.event_bus.raise_event(Event('TO_ACCUMULATOR', self.mask.get_array())))

        self.clear_button = ctk.CTkButton(self.frame, text='Очистить\nАккумулятор', width=80, command=self.clear,
                                          text_color='#7FFF00', fg_color='#000',
                                          hover_color='#006400', )
        self.clear_button.grid(row=0, column=0, padx=5, sticky='nsew')

        self.scroll = ctk.CTkScrollableFrame(self, orientation='vertical', height=490, width=205)
        self.scroll.grid(row=0, column=1, padx=5, pady=5)

        self.cages = []

        self.event_reactions['TO_ACCUMULATOR'] = lambda array: self.add_cage(array)

    def clear(self):
        for cage in self.cages:
            cage.grid_forget()
            del cage

        self.cages = []
        self.mask.set_array(np.zeros_like(self.mask.get_array()))

    def add_cage(self, array):
        cage = AccumulatorCage(self.scroll, len(self.cages), array, self.event_bus, lambda: self.refresh())
        cage.grid(pady=5)
        value = self.create_lambda(len(self.cages))
        cage.button_close.configure(command=lambda: self.delete_cage(value()))
        self.cages.append(cage)

        mask = self.mask.get_array()
        for cage in self.cages:
            mask = mask + cage.mask.get_array()
        self.mask.set_array(mask % (2 * np.pi))

    def delete_cage(self, number):
        cage = self.cages.pop(number)
        cage.grid_forget()
        cages = self.cages
        self.clear()
        mask = np.zeros_like(self.mask.get_array())
        for item in cages:
            self.add_cage(item.mask.get_array())
            mask = mask + item.mask.get_array()
        self.mask.set_array(mask % (2 * np.pi))

    def refresh(self):
        mask = np.zeros_like(self.mask.get_array())
        for item in self.cages:
            mask = mask + item.mask.get_array()
        self.mask.set_array(mask % (2 * np.pi))

    @staticmethod
    def create_lambda(x=None):
        return lambda: x


class AccumulatorCage(ctk.CTkFrame):
    def __init__(self, master, number, array, event_bus: EventBus, command):
        super().__init__(master, fg_color='#000', bg_color='#000')
        self.number = number
        self.event_bus = event_bus

        gray = event_bus.raise_request(Event('SLM_GRAY'))

        self.mask = MaskView(self, array, gray, small_res_x=160, small_res_y=100)
        self.mask.grid(row=0, column=0, rowspan=3)

        self.mask.add_menu_command('Отправить на SLM',
                                   lambda: self.event_bus.raise_event(Event('TO_SLM', self.mask.get_array())))
        self.mask.add_menu_command('Отправить в Атлас',
                                   lambda: self.event_bus.raise_event(Event('TO_ATLAS', self.mask.get_array())))
        self.mask.add_menu_command('Отправить в Аккумулятор',
                                   lambda: self.event_bus.raise_event(Event('TO_ACCUMULATOR', self.mask.get_array())))

        self.mask.post_commands.append(command)

        self.button_close = ctk.CTkButton(self, width=10, text='\u274C', fg_color='#F00')
        self.button_close.grid(row=0, column=1, sticky='n')
