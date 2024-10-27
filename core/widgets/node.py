import customtkinter as ctk


class Wire:
    def __init__(self, master, input_socket, output_socket, wire):
        self.master = master
        self.input_socket = input_socket
        self.output_socket = output_socket
        self.wire = wire

    def translate(self, value):
        self.output_socket.fill_value(value)

    def move(self):
        self.master.coords(self.wire, self.input_socket.x, self.input_socket.y, self.output_socket.x,
                           self.output_socket.y)


class Socket(ctk.CTkFrame):
    def __init__(self, master, node, canvas, name, height=10):
        super().__init__(master, fg_color='#000')

        self.node = node
        self.canvas = canvas
        self.name = name

        self.wire = None
        self.value = None
        self.is_filled = False

        self.ID = None

        self.height = height

        self.x = None
        self.y = None

    def move(self, event):
        self.canvas.move(self.ID, event.x, event.y)


class SocketIn(Socket):
    def __init__(self, master, node, canvas, name='IN', height=10):
        super().__init__(master, node, canvas, name, height)

        self.label = ctk.CTkLabel(self, text=f'\u23F5 {self.name}', fg_color='#000', height=self.height)
        self.label.grid(row=0, column=0, padx=5)
        self.label.update()
        self.label.update_idletasks()

    def move(self, event):
        super().move(event)
        self.x += event.x
        self.y += event.y
        self.canvas.ID_sockets_pos[str(self.ID)] = [self.x, self.y]
        if self.wire is not None:
            self.wire.move()

    def start(self):
        self.label.update()
        self.node.inputs_frame.update()
        radius = 7
        delta = self.node.inputs_frame.winfo_y() - self.node.label.winfo_height()
        abs_coord_x = self.canvas.start_x - radius
        abs_coord_y = self.canvas.start_y + self.node.label.winfo_height() + self.label.winfo_height() // 2 + self.winfo_y() + delta

        self.x = abs_coord_x
        self.y = abs_coord_y

        self.ID = self.canvas.create_oval(abs_coord_x - radius, abs_coord_y - radius, abs_coord_x + radius,
                                          abs_coord_y + radius, fill='#FFF', activefill="#0F0")

        self.canvas.sockets_in_id[str(self.ID)] = self

        self.canvas.ID_sockets_pos[str(self.ID)] = [self.x, self.y]

        self.canvas.tag_bind(self.ID, '<Button-1>', lambda event: self.canvas.start_connect(self, event))
        self.canvas.tag_bind(self.ID, '<B1-Motion>', self.canvas.move_connect)
        self.canvas.tag_bind(self.ID, '<ButtonRelease>', lambda event: self.canvas.end_connect(self, event))

    def fill_value(self, value):
        self.value = value
        self.is_filled = True
        self.node.fill_value(value, name=self.name)


class SocketOut(Socket):
    def __init__(self, master, node, canvas, name='OUT'):
        super().__init__(master, node, canvas, name)

        self.label = ctk.CTkLabel(self, text=f'{self.name} \u23F5', fg_color='#000', height=self.height)
        self.label.grid(row=0, column=0, padx=5)
        self.label.update()
        self.label.update_idletasks()

        self.ID = None

    def move(self, event):
        super().move(event)
        self.x += event.x
        self.y += event.y
        self.canvas.ID_sockets_pos[self.ID] = [self.x, self.y]
        if self.wire is not None:
            self.wire.move()

    def start(self):
        self.label.update()
        self.node.outputs_frame.update()
        radius = 7
        delta = self.node.outputs_frame.winfo_y() - self.node.label.winfo_height()
        abs_coord_x = self.canvas.start_x + self.node.label.winfo_width() + radius
        abs_coord_y = self.canvas.start_y + self.node.label.winfo_height() + self.label.winfo_height() // 2 + self.winfo_y() + delta

        self.x = abs_coord_x
        self.y = abs_coord_y

        self.ID = self.canvas.create_oval(abs_coord_x - radius, abs_coord_y - radius, abs_coord_x + radius,
                                          abs_coord_y + radius, fill='#FFF', activefill="#0F0")

        self.canvas.ID_sockets_pos[str(self.ID)] = [self.x, self.y]

        self.canvas.sockets_out_id[str(self.ID)] = self

        self.canvas.tag_bind(self.ID, '<Button-1>', lambda event: self.canvas.start_connect(self, event))
        self.canvas.tag_bind(self.ID, '<B1-Motion>', self.canvas.move_connect)
        self.canvas.tag_bind(self.ID, '<ButtonRelease>', lambda event: self.canvas.end_connect(self, event))

    def fill_value(self, value):
        if self.wire is not None:
            self.wire.translate(value)


class Node(ctk.CTkFrame):
    def __init__(self, master, name, x, y, width=100, height=100, color="#0000CD"):
        super().__init__(master, width=width, height=height, fg_color='grey20')

        self.color = color
        my_font = ctk.CTkFont(family="Helvetica", size=20,
                              weight="bold", slant="italic")
        self.label = ctk.CTkLabel(self, text=name, fg_color=color, text_color='#FFF', font=my_font)
        self.label.grid(row=0, column=0, columnspan=3,
                        sticky='nsew')

        self.inputs_frame = ctk.CTkFrame(self, width=50, fg_color='grey20', height=height)
        self.inputs_frame.grid(row=1, column=0, columnspan=1)

        self.center_frame = ctk.CTkFrame(self, width=50, fg_color='grey20', height=height)
        self.center_frame.grid(row=1, column=1, columnspan=1)

        self.outputs_frame = ctk.CTkFrame(self, width=50, fg_color='grey20', height=height)
        self.outputs_frame.grid(row=1, column=2, columnspan=1)

        self.input_sockets = []
        self.output_sockets = []

        self.label.bind('<B1-Motion>', self.move)

        self.ID = None
        self.state = None

        self.x = x
        self.y = y

        self.func_args = {}
        self.func_results = {}

        self.counter = 0

    def func(self, kwargs):
        pass

    def fill_value(self, value, name):
        self.func_args[name] = value
        self.counter += 1

        if self.counter == len(self.func_args):
            self.func_results = self.func(self.func_args)
            self.raise_results()

    def raise_results(self):
        for socket in self.output_sockets:
            socket.fill_value(self.func_results[socket.name])

    def move(self, event):
        self.master.move(self.ID, event.x, event.y)
        for item in self.input_sockets:
            item.move(event)

        for item in self.output_sockets:
            item.move(event)

    def start(self):
        for item in self.input_sockets:
            item.start()

        for item in self.output_sockets:
            item.start()

    def add_in(self, name='TEST'):
        socket = SocketIn(self.inputs_frame, self, self.master, name=name)
        socket.grid(pady=10, sticky='w')

        self.input_sockets.append(socket)

    def add_out(self, name='TEST'):
        socket = SocketOut(self.outputs_frame, self, self.master, name=name)
        socket.grid(pady=10, sticky='e')

        self.output_sockets.append(socket)


class TestNode(Node):
    def __init__(self, master, x, y):
        super().__init__(master, name='Test', x=x, y=y)

        self.add_in('yhhhho')

        self.add_out('UOOAD')
        self.add_out('UOOAD')
        self.add_out('UOOAD')


class PlusNode(Node):
    def __init__(self, master, x, y):
        super().__init__(master, name='+', x=x, y=y)

        self.add_in('first')
        self.add_in('second')

        self.add_out('sum')

    def func(self, args):
        return {'sum': args['first'] + args['second']}


class Source(Node):
    def __init__(self, master, arg_names, name, x, y, width=100, height=100, color="#00C000"):
        super().__init__(master, name, x, y, width, height, color)
        self.inputs_frame.grid_forget()
        self.center_frame.grid_forget()
        for item in arg_names:
            self.add_out(item)

    def raise_result(self, value):
        self.output_sockets[0].fill_value(value)


class NumSource(Source):
    def __init__(self, master, arg_names, name, x, y, width=100, height=100, color="#0000CD"):
        super().__init__(master, arg_names, name, x, y, width, height, color)

        self.value = 5

    def raise_result(self, value):
        self.output_sockets[0].fill_value(self.value)


class EndPoint(Node):
    def __init__(self, master, name, x, y, width=100, height=100, color="#0000CD"):
        super().__init__(master, name, x, y, width, height, color)
        self.outputs_frame.grid_forget()
        self.center_frame.grid_forget()
        self.add_in(name)

    def fill_value(self, value, name):
        self.master.results[name] = value


class NodeBoard(ctk.CTkCanvas):
    def __init__(self, master, bg="grey10", width=1000, height=800):
        super().__init__(master, bg=bg, width=width, height=height, bd=0, highlightthickness=0)

        self.sources = []
        self.nodes = []
        self.end_points = []
        self.results = {}

        self.start_x = 100
        self.start_y = 200

        self.width = width
        self.height = height

        self.sockets_in_id = {}
        self.sockets_out_id = {}

        self.ID_sockets_pos = {}

        self.add_cage()

        self.add_node(PlusNode)

        self.add_source(['5'], 'NUM', NumSource)
        self.add_source(['5'], 'NUM', NumSource)

        self.add_end('END')

        self.hold = False

        self.hold_x = None
        self.hold_y = None
        self.wire = None

        self.wires = {}

        self.hold_type = None

        self.master.bind("<z>", lambda event: self.execute(event, {'5': 5}))

    def execute(self, event, sources_dict):
        print('adad')
        for item in self.sources:
            item.raise_result(sources_dict[item.name])

        return self.results

    def start_connect(self, socket, event):
        self.hold = True
        self.hold_x = event.x
        self.hold_y = event.y
        self.wire = self.create_line(event.x, event.y, event.x, event.y, fill='#00FD00', width=5, activefill='#FFF')
        self.tag_lower(self.wire)

        if isinstance(socket, SocketOut) or isinstance(socket, Source):
            self.hold_type = 'SOURCE'
        else:
            self.hold_type = 'END'

    def move_connect(self, event):
        self.coords(self.wire, self.hold_x, self.hold_y, event.x - 10, event.y - 10)

    def end_connect(self, socket, event):
        ID = str(self.find_closest(event.x, event.y)[0])
        self.hold = False
        if ID in self.sockets_out_id and self.hold_type == 'END':

            wire = Wire(self, socket, self.sockets_out_id[ID], self.wire)
            socket.wire = wire
            self.sockets_out_id[ID].wire = wire
            self.coords(self.wire, self.hold_x, self.hold_y, self.ID_sockets_pos[ID][0],
                        self.ID_sockets_pos[ID][1])
            self.wires[self.wire] = wire
            self.tag_bind(self.wire, '<Button-3>',
                          lambda event: self.delete_wire(self.wire, socket, self.sockets_out_id[ID], event))
            print(self.wire)

        elif ID in self.sockets_in_id and self.hold_type == 'SOURCE':
            wire = Wire(self, self.sockets_in_id[ID], socket, self.wire)
            socket.wire = wire
            self.sockets_in_id[ID].wire = wire
            self.coords(self.wire, self.hold_x, self.hold_y, self.ID_sockets_pos[ID][0],
                        self.ID_sockets_pos[ID][1])
            self.wires[self.wire] = wire
            self.tag_bind(self.wire, '<Button-3>',
                          lambda event: self.delete_wire(self.sockets_in_id[ID], socket, event))

        else:
            self.delete(self.wire)
            self.wire = None

    def delete_wire(self, socket1, socket2, event):
        ID = self.find_closest(event.x, event.y)[0]
        self.delete(ID)

        socket1.wire = None
        socket2.wire = None

    def add_source(self, arg_names, name, class_type):
        node = class_type(self, arg_names, name, self.start_x, self.start_y)

        ID = self.create_window(self.start_x, self.start_y, window=node, anchor="nw")
        node.ID = ID

        self.sources.append(node)
        node.start()

    def add_end(self, name, color='#CD0000'):
        node = EndPoint(self, name, self.start_x, self.start_y, color=color)

        ID = self.create_window(self.start_x, self.start_y, window=node, anchor="nw")
        node.ID = ID

        self.end_points.append(node)
        node.start()

    def add_node(self, class_type):
        node = class_type(self, self.start_x, self.start_y)

        ID = self.create_window(self.start_x, self.start_y, window=node, anchor="nw")
        node.ID = ID

        self.nodes.append(node)
        node.start()

    def add_cage(self):
        CAGE = 100

        for i in range(1, self.height // CAGE):
            line = self.create_line(0, i * CAGE, self.width, i * CAGE, fill='#808080')
            self.tag_lower(line)

        for i in range(1, self.width // CAGE):
            line = self.create_line(i * CAGE, 0, i * CAGE, self.height, fill='#808080')
            self.tag_lower(line)


if __name__ == '__main__':
    app = ctk.CTk()
    board = NodeBoard(app)
    board.grid()
    app.mainloop()
