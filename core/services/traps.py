from core.services.service import Service

from core.event_bus import Event

import customtkinter as ctk
import tksheet

import matplotlib.pyplot as plt

from functools import lru_cache
import numpy as np
import numba


class TrapsEditor(Service):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, name='Редактор Ловушек', width=500, height=1, **kwargs)

        self.top_frame.grid_forget()

        self.sheet = tksheet.Sheet(self.frame, theme='dark green', width=520,
                                   headers=['X', 'Y', 'Z', 'W'])

        self.sheet.grid(padx=5, pady=5, sticky='nsew')

        self.sheet.enable_bindings()

        self.sheet.disable_bindings('copy', 'rc_insert_column', 'paste', 'cut', 'Delete', 'Edit cell', 'Delete columns')

        self.sheet.set_options(insert_row_label='Добавить Ловушку')
        self.sheet.set_options(delete_rows_label='Удалить Ловушку')
        self.sheet.set_options(insert_rows_above_label='Добавить ловушку \u25B2')
        self.sheet.set_options(insert_rows_below_label='Добавить ловушку \u25BC')
        self.sheet.set_options(delete_rows_label='Удалить Ловушку')

        frame = ctk.CTkFrame(self.frame)
        frame.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        frame_xc = ctk.CTkFrame(frame)
        frame_xc.grid(row=0, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_xc, text='X-Координата Центра (мкм) :', bg_color='#000').grid(row=0, column=0)

        self.x_c = ctk.CTkEntry(frame_xc, width=50, bg_color='#1E90FF')
        self.x_c.insert(0, '0')
        self.x_c.grid(row=0, column=1)

        frame_yc = ctk.CTkFrame(frame)
        frame_yc.grid(row=1, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_yc, text='Y-Координата Центра (мкм) :', bg_color='#000').grid(row=0, column=0)

        self.y_c = ctk.CTkEntry(frame_yc, width=50, bg_color='#1E90FF')
        self.y_c.insert(0, '0')
        self.y_c.grid(row=0, column=1)

        frame_zc = ctk.CTkFrame(frame)
        frame_zc.grid(row=2, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_zc, text='Z-Координата Центра (мкм) :', bg_color='#000').grid(row=0, column=0)

        self.z_c = ctk.CTkEntry(frame_zc, width=50, bg_color='#1E90FF')
        self.z_c.insert(0, '0')
        self.z_c.grid(row=0, column=1)

        frame_angle = ctk.CTkFrame(frame)
        frame_angle.grid(row=3, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_angle, text='Угол Поворота (град.) :', bg_color='#000').grid(row=0, column=0)

        self.angle = ctk.CTkEntry(frame_angle, width=50, bg_color='#1E90FF')
        self.angle.insert(0, '0')
        self.angle.grid(row=0, column=1)

        frame_w = ctk.CTkFrame(frame)
        frame_w.grid(row=4, column=0, padx=5, pady=5)
        ctk.CTkLabel(frame_w, text='Вес :', bg_color='#000').grid(row=0, column=0)
        self.w = ctk.CTkEntry(frame_w, width=50, bg_color='#1E90FF')
        self.w.insert(0, '1')
        self.w.grid(row=0, column=1)

        ctk.CTkButton(frame, text='Удалить ловушки', fg_color='#000', text_color='#1E90FF',
                      command=self.delete_all).grid(row=5, column=0, padx=5)

        ctk.CTkButton(frame, text='Карта ловушек', fg_color='#000', text_color='#1E90FF',
                      command=self.plot_traps).grid(row=5, column=1, padx=5)

        notebook = ctk.CTkTabview(frame, segmented_button_selected_color='#000',
                                  text_color='#1E90FF', segmented_button_selected_hover_color='#191970', width=200,
                                  height=150)

        notebook.grid(row=0, column=1, rowspan=5)
        notebook.add('Массив')

        frame_xn = ctk.CTkFrame(notebook.tab('Массив'))
        frame_xn.grid(row=0, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_xn, text='Число по X:', bg_color='#000').grid(row=0, column=0)

        self.x_n = ctk.CTkEntry(frame_xn, width=50, bg_color='#1E90FF')
        self.x_n.insert(0, '5')
        self.x_n.grid(row=0, column=1)

        frame_yn = ctk.CTkFrame(notebook.tab('Массив'))
        frame_yn.grid(row=1, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_yn, text='Число по Y:', bg_color='#000').grid(row=0, column=0)

        self.y_n = ctk.CTkEntry(frame_yn, width=50, bg_color='#1E90FF')
        self.y_n.insert(0, '5')
        self.y_n.grid(row=0, column=1)

        frame_zn = ctk.CTkFrame(notebook.tab('Массив'))
        frame_zn.grid(row=2, column=0, padx=5, pady=5)

        ctk.CTkLabel(frame_zn, text='Число по Z:', bg_color='#000').grid(row=0, column=0)

        self.z_n = ctk.CTkEntry(frame_zn, width=50, bg_color='#1E90FF')
        self.z_n.insert(0, '1')
        self.z_n.grid(row=0, column=1)

        frame_xd = ctk.CTkFrame(notebook.tab('Массив'))
        frame_xd.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(frame_xd, text='Период по X:', bg_color='#000').grid(row=0, column=0)

        self.x_d = ctk.CTkEntry(frame_xd, width=50, bg_color='#1E90FF')
        self.x_d.insert(0, '200')
        self.x_d.grid(row=0, column=1)

        frame_yd = ctk.CTkFrame(notebook.tab('Массив'))
        frame_yd.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(frame_yd, text='Период по Y:', bg_color='#000').grid(row=0, column=0)

        self.y_d = ctk.CTkEntry(frame_yd, width=50, bg_color='#1E90FF')
        self.y_d.insert(0, '200')
        self.y_d.grid(row=0, column=1)

        frame_zd = ctk.CTkFrame(notebook.tab('Массив'))
        frame_zd.grid(row=2, column=1, padx=5, pady=5)

        ctk.CTkLabel(frame_zd, text='Период по Z:', bg_color='#000').grid(row=0, column=0)

        self.z_d = ctk.CTkEntry(frame_zd, width=50, bg_color='#1E90FF')
        self.z_d.insert(0, '200')
        self.z_d.grid(row=0, column=1)

        ctk.CTkButton(notebook.tab('Массив'), text='Добавить', fg_color='#000', text_color='#1E90FF',
                      command=self.add_array).grid(row=3, column=0, columnspan=2)

        self.request_reactions['TRAPS_SPECS'] = lambda: self.prepare_specs()

        self.event_reactions['NEW_WEIGHTS'] = lambda weights: self.new_weights(weights)

    def new_weights(self, weights):
        for i in range(self.sheet.get_total_rows()):
            spec = self.sheet[i].data
            self.sheet[i].data = [spec[0], spec[1], spec[2], str(weights[i])]

    def plot_traps(self):
        x = []
        y = []
        w = []
        name = []
        for i in range(self.sheet.get_total_rows()):
            spec = self.sheet[i].data
            if spec[0] != '' and spec[1] != '' and spec[2] != '' and spec[3] != '':
                x.append(float(spec[0]))
                y.append(float(spec[1]))
                w.append(float(spec[3]))
                name.append(i + 1)

        plt.style.use('dark_background')
        plt.scatter(x, y, c=w, alpha=0.7, cmap='hot', vmin=0, vmax=max(w))
        plt.colorbar(cmap='hot')
        plt.show()

    def prepare_specs(self):
        traps_x = []
        traps_y = []
        weights = []

        for i in range(self.sheet.get_total_rows()):
            spec = self.sheet[i].data
            if spec[0] != '' and spec[1] != '' and spec[2] != '' and spec[3] != '':
                traps_x.append(float(spec[0]) * 10 ** -6)
                traps_y.append(float(spec[1]) * 10 ** -6)
                weights.append(float(spec[3]))

        return np.asarray(traps_x), np.asarray(traps_y), np.asarray(weights)

    def add_array(self):
        x_c = float(self.x_c.get())
        y_c = float(self.y_c.get())
        z_c = float(self.z_c.get())

        x_n = int(self.x_n.get())
        y_n = int(self.y_n.get())
        z_n = int(self.z_n.get())

        x_d = float(self.x_d.get())
        y_d = float(self.y_d.get())
        z_d = float(self.z_d.get())

        x_line = [x_c - x_d * (x_n - 1) / 2 + x_d * i for i in range(x_n)]
        y_line = [y_c - y_d * (y_n - 1) / 2 + y_d * i for i in range(y_n)]
        z_line = [z_c - z_d * (z_n - 1) / 2 + z_d * i for i in range(z_n)]

        w = float(self.w.get())
        for z in z_line:
            for x in x_line:
                for y in y_line:
                    self.sheet.insert_row([x, y, z, w])

    def delete_all(self):
        for i in range(self.sheet.get_total_rows()):
            self.sheet.del_row(0)

    def delete(self):
        current_selection = self.sheet.get_currently_selected()
        if current_selection:
            box = current_selection.row
            self.sheet.delete_row(box)


@lru_cache
def calc_holo(x, y, z, wave, focus, d, width, height):
    x *= 10 ** -6
    y *= 10 ** -6
    z *= 10 ** -6

    _x = np.linspace(- width // 2 * d, width // 2 * d, width)
    _y = np.linspace(-height // 2 * d, height // 2 * d, height)

    _x, _y = np.meshgrid(_x, _y)

    sphere = np.pi * z / wave / focus / focus * (_x * _x + _y * _y)
    lattice = 2 * np.pi / wave / focus * (x * _x + y * _y)
    holo = (sphere + lattice) % (2 * np.pi)
    return holo
