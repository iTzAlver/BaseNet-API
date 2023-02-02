# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import multiprocessing as mp
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image
from ..database import BaseNetDatabase
from ..deeplearning import BaseNetModel
from .visualizer import BaseNetCVVisualizer
from ..__special__ import __cviz_ico_location__, __version__

COLOR_STYLES = {'blue': '#8685cb', 'pink': '#c539b7', 'yellow': '#c8c963', 'gray': '#BBBBBB',
                'red': '#c42717', 'orange': '#cc5500', 'black': '#111111', 'green': '#00ff00', 'white': '#cccccc'}


# -----------------------------------------------------------
def basenet_cv_gui():
    process = mp.Process(target=__gui)
    process.start()
    return process


def __gui() -> None:
    logging.info('[+] BaseNetCVGUI: Started BaseNetCVGUI in other process.')
    root_node = tk.Tk()
    MainWindow(root_node)
    try:
        root_node.iconbitmap(__cviz_ico_location__)
    except tk.TclError:
        alternative_bitmap = tk.PhotoImage(f'{__cviz_ico_location__.split(".")[0]}.xbm')
        root_node.call('wm', 'iconphoto', root_node._w, alternative_bitmap)
    root_node.configure()
    root_node.mainloop()
    logging.info('[-] BaseNetCVGUI: Finished BaseNetCVGUI in other process.')


class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("BaseNetCVGUI")
        self.master.geometry('620x870')
        self.master.minsize(620, 870)
        self.master.maxsize(620, 870)
        self.master.configure(bg='white')
        self.color_styles = COLOR_STYLES

        self.database = None
        self.model = None
        self.access = 'test'
        self.visualizer = None
        self.images = list()
        self.index = None

        self.in_canvas = None
        self.in_toolbar = None
        self.out_canvas = None
        self.out_toolbar = None
        # - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
        #                        FRAMES                             #
        # - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
        self.title_label = tk.Label(self.master, text=f'BaseNetCVGUI v{__version__}', fg='black', bg='white',
                                    font='Fixedsys 31 bold')
        self.title_label.place(x=50, y=5)
        self.insert_database_button = tk.Button(self.master, text='SELECT DATABASE',
                                                command=self.b_select_database, font='Bahnschrift 8 bold', fg='black',
                                                width=15, bg=self.color_styles['green'])
        self.insert_database_button.place(x=25, y=70)
        self.insert_basenetm_button = tk.Button(self.master, text='SELECT MODEL',
                                                command=self.b_select_model, font='Bahnschrift 8 bold', fg='black',
                                                width=15, bg=self.color_styles['orange'])
        self.insert_basenetm_button.place(x=225, y=70)
        self.insert_basenetm_button = tk.Button(self.master, text='MAP OUTPUT',
                                                command=self.b_map_output, font='Bahnschrift 8 bold', fg='black',
                                                width=15, bg=self.color_styles['red'])
        self.insert_basenetm_button.place(x=425, y=70)

        self.insert_nextings_button = tk.Button(self.master, text='>',
                                                command=self.b_next, font='Bahnschrift 8 bold', fg='black',
                                                width=5, bg=self.color_styles['blue'])
        self.insert_nextings_button.place(x=535, y=120)
        self.insert_beforing_button = tk.Button(self.master, text='<',
                                                command=self.b_before, font='Bahnschrift 8 bold', fg='black',
                                                width=5, bg=self.color_styles['blue'])
        self.insert_beforing_button.place(x=25, y=120)

        self.access_train = tk.Checkbutton(self.master, text='Take data from \t"Train" \t\tdataset',
                                           command=self.c_select_train, bg='white')
        self.access_train.place(x=175, y=110)
        self.access_val = tk.Checkbutton(self.master, text='Take data from \t"Validation" \tdataset',
                                         command=self.c_select_validation, bg='white')
        self.access_val.place(x=175, y=130)
        self.access_test = tk.Checkbutton(self.master, text='Take data from \t"Test" \t\tdataset',
                                          command=self.c_select_test, bg='white')
        self.access_test.place(x=175, y=150)
        self.access_test.select()
        # - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
        #                        LABEL FRAMES                       #
        # - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
        self.__label_frame_0 = tk.LabelFrame(self.master, width=610, height=600)
        self.__label_frame_0.place(x=5, y=180)
        self.__label_frame_1 = tk.LabelFrame(self.master, width=610, height=80)
        self.__label_frame_1.place(x=5, y=785)

    def b_select_database(self):
        dbpath = filedialog.askopenfilename(filetypes=[('Database files', '*.db')])
        if dbpath:
            self.database = BaseNetDatabase.load(dbpath)
            if self.visualizer is None:
                self.visualizer = BaseNetCVVisualizer(self.database)
            else:
                self.visualizer.link_database(self.database)
                self.visualizer.set_access(self.access)
        if len(getattr(self.database, f'x{self.access}')) == 0:
            self.access = 'train'
            self.visualizer.set_access(self.access)
            self.access_train.select()
            self.access_test.deselect()
            self.access_val.deselect()

    def b_select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[('Database files', '*.h5')])
        if model_path:
            self.model = BaseNetModel.load(model_path)
            if self.visualizer is not None:
                self.visualizer.link_model(self.model)
            else:
                self.visualizer = BaseNetCVVisualizer(self.model.breech[0], model=self.model)
            self.visualizer.set_access(self.access)

    def b_map_output(self):
        output = self.model.predict([self.images[self.index][-1]])[0]
        img_output = Image.fromarray((output * 255).T)
        self.draw_out(img_output)

    def b_next(self):
        if self.index is not None:
            self.index += 1
        elif self.visualizer is not None:
            self.index = 0
            self.images.append(self.visualizer.get())
        else:
            return
        if self.index == len(self.images):
            self.images.append(self.visualizer.get())
        self.draw_in(self.images[self.index][0], label=self.images[self.index][1])

    def b_before(self):
        if self.index is not None:
            self.index -= 1
            if self.index < 0:
                self.index = len(self.images) - 1
        elif self.visualizer is not None:
            self.index = 0
            self.images.append(self.visualizer.get())
        else:
            return
        self.draw_in(self.images[self.index][0], label=self.images[self.index][1])

    def draw_in(self, in_fig, label):
        _in_fig = plt.figure(figsize=(7.20, 7.4), dpi=75)
        plt.imshow(in_fig)
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.title(f'Output from image number {self.index} with label: {label}')
        if self.in_canvas is not None:
            self.in_canvas.get_tk_widget().pack_forget()
            self.in_toolbar.destroy()
        self.in_canvas = FigureCanvasTkAgg(_in_fig, master=self.__label_frame_0)
        self.in_canvas.draw()
        self.in_toolbar = NavigationToolbar2Tk(self.in_canvas, self.__label_frame_0)
        self.in_toolbar.update()
        self.in_canvas.get_tk_widget().pack()

    def draw_out(self, out_fig):
        _out_fig = plt.figure(figsize=(1.3, 1.3), dpi=75)
        plt.imshow(out_fig)
        plt.title(f'Output from image number {self.index}')
        if self.out_canvas is not None:
            self.out_canvas.get_tk_widget().pack_forget()
            self.out_canvas.destroy()
        self.out_canvas = FigureCanvasTkAgg(_out_fig, master=self.__label_frame_1)
        self.out_canvas.draw()
        self.out_toolbar = NavigationToolbar2Tk(self.out_canvas, self.__label_frame_1)
        self.out_toolbar.update()
        self.out_canvas.get_tk_widget().pack()

    def c_select_train(self):
        self.access = 'train'
        self.access_test.deselect()
        self.access_val.deselect()
        if self.visualizer is not None:
            self.visualizer.set_access(self.access)

    def c_select_validation(self):
        self.access = 'val'
        self.access_train.deselect()
        self.access_test.deselect()
        if self.visualizer is not None:
            self.visualizer.set_access(self.access)

    def c_select_test(self):
        self.access = 'test'
        self.access_train.deselect()
        self.access_val.deselect()
        if self.visualizer is not None:
            self.visualizer.set_access(self.access)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
