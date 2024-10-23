from core.services.device import Device

import customtkinter as ctk

import cv2


class Filter(ctk.CTkFrame):
    def __init__(self, master, name):
        super().__init__(master, fg_color='#000', bg_color='#000')

        self.label = ctk.CTkLabel(self, text=name, fg_color='#000')
        self.label.grid(row=0, column=0)

        self.check_var = ctk.StringVar(value="on")
        self.checkbox = ctk.CTkCheckBox(self, text="Применить",
                                        variable=self.check_var, onvalue="on", offvalue="off")

        self.checkbox.grid(row=1, column=0, columnspan=2)
        self.checkbox.deselect()

        self.entry = ctk.CTkEntry(self, width=50, bg_color='#1E90FF')
        self.entry.insert(0, '20')
        self.entry.grid(row=0, column=1)

    def apply(self, array):
        if self.check_var.get() == 'on':
            self.do_filter(array)

    def do_filter(self, array):
        pass


class Thresh(Filter):
    def __init__(self, master):
        super().__init__(master, name='Порог')

    def apply(self, array):
        value = int(self.entry.get())
        _, array = cv2.threshold(array, value, 255, cv2.THRESH_TOZERO)

        return array


class Camera(Device):
    def __init__(self, master, ):
        super().__init__(master, name='Камера')

        notebook = ctk.CTkTabview(self.frame, segmented_button_selected_color='#000',
                                  text_color='#7FFF00', segmented_button_selected_hover_color='#006400', width=180)
        notebook.grid(row=0)
        notebook.add('ROI')
        notebook.add('Фильтр')

        ctk.CTkButton(self.frame, text='Тест Камеры:\nСнимок', text_color='#7FFF00', fg_color='#000',
                      hover_color='#006400', command=self.show_shot).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.frame, text='Тест Камеры:\nПоток', text_color='#7FFF00', fg_color='#000',
                      hover_color='#006400', command=self.show_stream).grid(row=2, column=0, padx=5, pady=5)

        frame = ctk.CTkFrame(notebook.tab('ROI'))
        frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        frame_x_left = ctk.CTkFrame(frame)
        frame_x_left.grid(row=0, column=0, padx=5)
        ctk.CTkLabel(frame_x_left, text='\u21F1 x:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.x_left = ctk.CTkEntry(frame_x_left, width=45, bg_color='#FF4500')
        self.x_left.insert(0, '200')
        self.x_left.grid(row=0, column=1)

        frame_y_left = ctk.CTkFrame(frame)
        frame_y_left.grid(row=1, column=0, padx=5)
        ctk.CTkLabel(frame_y_left, text='\u21F1 y:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.y_left = ctk.CTkEntry(frame_y_left, width=45, bg_color='#FF4500')
        self.y_left.insert(0, '200')
        self.y_left.grid(row=0, column=1)

        frame_x_right = ctk.CTkFrame(frame)
        frame_x_right.grid(row=0, column=1, padx=5)
        ctk.CTkLabel(frame_x_right, text='\u21F2 x:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)

        self.x_right = ctk.CTkEntry(frame_x_right, width=45, bg_color='#FF4500')
        self.x_right.insert(0, '300')
        self.x_right.grid(row=0, column=1)

        frame_y_right = ctk.CTkFrame(frame)
        frame_y_right.grid(row=1, column=1, padx=5)
        ctk.CTkLabel(frame_y_right, text='\u21F2 y:', fg_color='#000',
                     text_color='#FFF').grid(row=0, column=0)
        self.y_right = ctk.CTkEntry(frame_y_right, width=45, bg_color='#FF4500')
        self.y_right.insert(0, '300')
        self.y_right.grid(row=0, column=1)

        ctk.CTkButton(notebook.tab('ROI'), text='Тест ROI:\nСнимок', text_color='#FF4500', fg_color='#000',
                      hover_color='#DC143C', command=self.show_roi_shot).grid(row=3, column=0, padx=5, pady=5)
        ctk.CTkButton(notebook.tab('ROI'), text='Тест ROI:\nПоток', text_color='#FF4500', fg_color='#000',
                      hover_color='#DC143C', command=self.show_roi_stream).grid(row=4, column=0, padx=5, pady=5)

        ctk.CTkButton(notebook.tab('ROI'), text='Выбрать ROI', text_color='#FF4500', fg_color='#000',
                      hover_color='#DC143C', command=self.pick_roi).grid(row=5, column=0, padx=5, pady=5)

        self.cap = None

        self.request_reactions['TAKE_SHOT'] = lambda: self.take_shot()
        self.request_reactions['TAKE_ROI_SHOT'] = lambda: self.take_roi_shot()

        self.request_reactions['TAKE_ROI'] = lambda:(int(self.x_left.get()), int(self.y_left.get()), int(self.x_right.get()), int(self.y_right.get()))

        self.scroll = ctk.CTkScrollableFrame(notebook.tab('Фильтр'), width=100)
        self.scroll.grid()

        self.filters = []

        thresh = Thresh(self.scroll)
        thresh.grid()

        self.filters.append(thresh)

    def __del__(self):
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()

    def pick_roi(self):
        image = self.take_shot()

        # Select ROI
        r = cv2.selectROI("select the area", image)

        x_left = int(r[0])
        y_left = int(r[1])
        x_right = int(r[2]) + int(r[0])
        y_right = int(r[3]) + int(r[1])

        print(x_left, y_left, x_right, y_right)

        self.x_left.delete(0, 'end')
        self.x_left.insert(0, str(x_left))

        self.y_left.delete(0, 'end')
        self.y_left.insert(0, str(y_left))

        self.x_right.delete(0, 'end')
        self.x_right.insert(0, str(x_right))

        self.y_right.delete(0, 'end')
        self.y_right.insert(0, str(y_right))

        cv2.destroyWindow('select the area')

    def take_shot(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(int(self.port.get()), cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for f in self.filters:
            gray = f.apply(gray)

        return gray

    def show_shot(self):
        gray = self.take_shot()
        cv2.imshow('Camera Test', gray)

    def show_stream(self):

        while True:

            self.show_shot()
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyWindow('Camera Test')

    def take_roi_shot(self):
        gray = self.take_shot()

        x_left = int(str(self.x_left.get()))
        y_left = int(str(self.y_left.get()))

        x_right = int(str(self.x_right.get()))
        y_right = int(str(self.y_right.get()))

        return gray[y_left:y_right, x_left:x_right]

    def show_roi_shot(self):
        gray = self.take_roi_shot()
        cv2.imshow('Camera Test', gray)

    def show_roi_stream(self):

        while True:

            self.show_roi_shot()
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyWindow('Camera Test')
