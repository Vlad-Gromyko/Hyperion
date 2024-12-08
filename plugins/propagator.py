

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

from PIL import Image
import LightPipes as lp


class Plugin(AbstractPlugin):
    def __init__(self, master, sub_master, *args, **kwargs):
        super().__init__(master, sub_master, *args, name='Моделирование', width=500, height=40, **kwargs)

        self.top_frame.grid_forget()

        self.slm_grid_dim = None
        self.slm_grid_size = None

        self.camera_grid_dim = None
        self.camera_grid_size = None

        self.holo = None

        self.field = None

        self.field = None

        self.gauss_waist = 1 * 10 ** -3

        self.result = None

    def start(self):
        res_x = self.event_bus.raise_request(Event('SLM_WIDTH'))
        res_y = self.event_bus.raise_request(Event('SLM_HEIGHT'))

        pixel_slm = self.event_bus.raise_request(Event('SLM_PITCH'))
        wave = self.event_bus.raise_request(Event('WAVE'))

        self.slm_grid_dim = max(res_x, res_y)
        self.slm_grid_size = self.slm_grid_dim * pixel_slm

        self.camera_grid_dim = 1280
        self.camera_grid_size = self.camera_grid_dim * 3 * 10 ** -6 / 3

        self.field = lp.Begin(self.slm_grid_size, wave,
                              self.slm_grid_dim)

        self.field = lp.GaussBeam(self.field, self.gauss_waist)

        self.event_reactions['PROPAGATE'] = lambda a: self.propagate()

        self.request_reactions['PROPAGATED'] = lambda: self.result

    def propagate(self):
        focus = self.event_bus.raise_request(Event('FOCUS'))

        holo = self.event_bus.raise_request(Event('MASK_ON_SLM'))

        holo = self.holo_box(holo)
        field = lp.SubPhase(self.field, holo)
        field = lp.Lens(field, focus)
        field = lp.Forvard(field, focus)

        field = lp.Interpol(field, self.camera_grid_size,
                            self.camera_grid_dim)
        self.result = lp.Intensity(field)

        return self.result

    @staticmethod
    def holo_box(holo):
        rows, cols = holo.shape

        size = max(rows, cols)

        square_array = np.zeros((size, size), dtype=holo.dtype)

        row_offset = (size - rows) // 2
        col_offset = (size - cols) // 2

        square_array[row_offset:row_offset + rows, col_offset:col_offset + cols] = holo

        return square_array
