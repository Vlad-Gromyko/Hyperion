import customtkinter as ctk

from core.event_bus import EventBus

from core.services.camera import Camera
from core.services.slm import SLM
from core.services.traps import TrapsEditor
from core.services.lab import Lab
from core.services.atlas import Atlas
from core.services.accumulator import Accumulator
from core.services.progress import Progress

import numpy as np

import importlib
import importlib.util
import os.path
from os import listdir
from os.path import isfile, join


class Splash( ctk.CTkToplevel):
   def __init__(self, parent):
       ctk.CTkToplevel.__init__(self, parent)
       self.title("Splash")
       self.update()

class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        WIDTH = 1920
        HEIGHT = 1200
        GRAY = 255

        self.title('Hyperion 0.1')

        self.event_bus = EventBus()


        self.left_notebook = ctk.CTkTabview(self, width=100, height=200, segmented_button_selected_color='#000',
                                            text_color='#7FFF00', segmented_button_selected_hover_color='#006400')
        self.left_notebook.grid(row=0, column=0, padx=5, pady=5)

        self.left_notebook.add('Камера')
        self.left_notebook.add('SLM')

        self.camera = Camera(self.left_notebook.tab('Камера'))
        self.camera.grid()

        self.event_bus.register(self.camera)

        self.slm = SLM(self.left_notebook.tab('SLM'), WIDTH, HEIGHT, GRAY)
        self.slm.grid()

        self.event_bus.register(self.slm)

        self.center_notebook = ctk.CTkTabview(self, segmented_button_selected_color='#000',
                                              text_color='#7FFF00', segmented_button_selected_hover_color='#006400',
                                              width=500, height=300)
        self.center_notebook.add('Редактор Ловушек')
        self.center_notebook.grid(row=0, column=1, padx=5, pady=5, rowspan=2)

        self.center_notebook.add('Плагин')

        self.traps = TrapsEditor(self.center_notebook.tab('Редактор Ловушек'))
        self.traps.grid()

        self.event_bus.register(self.traps)

        self.notebook = ctk.CTkTabview(self, width=200, height=200, segmented_button_selected_color='#000',
                                       text_color='#7FFF00', segmented_button_selected_hover_color='#006400')
        self.notebook.add('Линза и Лазер')
        self.notebook.grid(row=1, column=0, padx=5, pady=5)
        self.lab = Lab(self.notebook.tab('Линза и Лазер'))
        self.lab.grid()

        self.event_bus.register(self.lab)

        notebook_right = ctk.CTkTabview(self, width=250, height=600, segmented_button_selected_color='#000',
                                        text_color='#7FFF00', segmented_button_selected_hover_color='#006400')
        notebook_right.add('Атлас')
        notebook_right.grid(row=0, column=2, padx=5, pady=5, rowspan=2)

        self.atlas = Atlas(notebook_right.tab('Атлас'))
        self.atlas.grid()

        self.event_bus.register(self.atlas)

        notebook_right.add('Аккумулятор')


        self.accumulator = Accumulator(notebook_right.tab('Аккумулятор'), WIDTH, HEIGHT, GRAY)
        self.accumulator.grid()

        self.event_bus.register(self.accumulator)

        self.accumulator.start()

        self.notebook.add('Плагины')

        self.plugins = []

        mypath = os.getcwd() + r'\plugins'
        self.apply_plugins(mypath)

        frame = ctk.CTkFrame(self)
        frame.grid(row=2, columnspan=3)
        self.progress = Progress(frame)
        self.progress.grid(row=0, column=0,)
        self.event_bus.register(self.progress)


    def apply_plugins(self, dir_path):
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        for item in files:
            self.dynamic_import(os.path.join(dir_path, item))

    def dynamic_import(self, path):
        module = self.import_module_from_path(path)
        plugin = module.Plugin(self.center_notebook.tab('Плагин'), self.notebook.tab('Плагины'))

        self.plugins.append(plugin)
        self.event_bus.register(plugin)
        plugin.start()

    def import_module_from_path(self, path: str):
        file_path = path

        spec = importlib.util.spec_from_file_location(f'plugin__{len(self.plugins)}', file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
