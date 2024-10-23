import customtkinter as ctk
from tkinter.filedialog import askdirectory

import numpy as np
from PIL import Image
import tkinter as tk
from tkinter.filedialog import asksaveasfilename, askopenfilename

import matplotlib.pyplot as plt


class MaskView(ctk.CTkFrame):
    def __init__(self, master, array: np.ndarray, slm_gray_edge: int = 224,
                 small_res_x: int = 160, small_res_y: int = 100,
                 *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.__small_res_x = small_res_x
        self.__small_res_y = small_res_y

        self.__slm_gray_edge = slm_gray_edge

        self.label = ctk.CTkLabel(self, text='')
        self.label.grid(padx=5, pady=5)

        self.__array = array

        self.menu = tk.Menu(self, tearoff=0)

        self.menu.add_command(label='Сохранить BMP', command=self.save)
        self.menu.add_command(label='Загрузить', command=self.load)
        self.menu.add_command(label='Случайный Шум', command=self.noise)
        self.menu.add_command(label='2D', command=self.plot2d)

        self.label.bind('<Button-3>', self.right_click)

        self.post_commands = []

        self.set_array(array)

    def load(self):
        filepath = askopenfilename()
        if filepath != "":
            img = Image.open(filepath).convert('L')

            w, h = np.shape(self.__array)
            img = img.resize((h, w))
            img = np.asarray(img)

            img = img / np.max(img) * 2 * np.pi
            self.set_array(img)

    def plot2d(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.__array, cmap='hot')
        ax.set_title('Маска')

        fig.colorbar(im, ax=ax, )

        plt.show()

    def add_menu_command(self, label: str, command):
        self.menu.add_command(label=label, command=command)

    def right_click(self, event):
        self.menu.post(event.x_root, event.y_root)

    def noise(self):
        size = self.__array.shape
        np.random.seed(42)
        phs = np.random.uniform(low=0, high=2 * np.pi, size=size)
        self.set_array(phs)

    def to_pixels(self, array):
        result = np.asarray(array / 2 / np.pi * self.__slm_gray_edge, dtype='uint8')
        return result

    def get_pixels(self):
        return self.to_pixels(self.__array)

    def save(self):
        try:
            save = asksaveasfilename(initialfile='new_mask.bmp',
                                     filetypes=[("bmp", '*.bmp')])
            if save != '':
                array = self.to_pixels(self.__array)
                image = Image.fromarray(array)
                image.save(save)

        except ValueError:
            pass

    def set_array(self, array: np.ndarray):
        self.__array = array
        image_to_show = self.to_pixels(array)
        photo = ctk.CTkImage(light_image=Image.fromarray(image_to_show),
                             size=(self.__small_res_x, self.__small_res_y))
        self.label.configure(image=photo)

        for command in self.post_commands:
            command()

    def get_array(self):
        return self.__array

    def set_gray_edge(self, value: int = 224):
        if 0 < value < 256:
            self.__slm_gray_edge = value
            self.set_array(self.__array)

    def set_small_res_x(self, value: int = 120):
        if 0 < value:
            self.__small_res_x = value
            self.set_array(self.__array)

    def set_small_res_y(self, value: int = 120):
        if 0 < value:
            self.__small_res_y = value
            self.set_array(self.__array)


if __name__ == '__main__':
    app = ctk.CTk()
    app.geometry("200x200")

    x = np.linspace(396, -396, 792)
    y = np.linspace(-300, 300, 600)
    x, y = np.meshgrid(x, y)

    phi = np.atan2(y, x) + np.pi

    image = MaskView(master=app, array=phi)
    image.grid()

    app.mainloop()
