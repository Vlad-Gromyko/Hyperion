import os.path
import customtkinter as ctk
import importlib
import importlib.util
from os import listdir
from os.path import isfile, join


class PluginPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, plugins_frame, slm, atlas, accumulator, translator, database=None, *args, **kwargs):
        super().__init__(*args, master=master, orientation='vertical', **kwargs)
        self.plugins = []
        self.plugins_frame = plugins_frame

        self.slm = slm
        self.atlas = atlas
        self.accumulator = accumulator
        self.translator = translator
        self.database = database

    def apply_plugins(self, dir_path):
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        for item in files:
            self.dynamic_import(os.path.join(dir_path, item))

    def dynamic_import(self, path):
        module = self.import_module_from_path(path)
        plugin = module.Plugin(self.plugins_frame, self, self.slm, self.atlas, self.accumulator, self.translator,
                               self.database)
        self.plugins.append(plugin)

    def import_module_from_path(self, path: str):
        file_path = path

        spec = importlib.util.spec_from_file_location(f'plugin__{len(self.plugins)}', file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def close_all(self):
        for item in self.plugins:
            item.frame.grid_forget()
            item.toggled.frame.grid_forget()
            item.toggled.opened = False
