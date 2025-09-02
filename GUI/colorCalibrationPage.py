import tkinter as tk

from GUI.videoFrame import VideoFrame

class ColorCalibrationPage(VideoFrame):

    @staticmethod
    def __rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    def __init__(self, root: tk.Tk, camera_index: int, next_page_listener: callable):
        super().__init__(root, camera_index)
        self.red_selected = (0, 0, 0)
        self.green_selected = (0, 0, 0)
        self.green_selecting = False
        self.red_selecting = False
        self.next_page_listener = next_page_listener

        self.video_label.bind("<Button-1>", self.__on_mouse_click)
        self.start()

        self.canvas = tk.Canvas(root, width=120, height=200)
        self.canvas.pack()
        self.red_oval = self.canvas.create_oval(0, 0, 50, 50, fill='red')
        self.green_oval = self.canvas.create_oval(0, 60, 50, 110, fill='green')

        select_green_button = tk.Button(root, text="Select Green", command=self.__on_select_green)
        select_green_button.pack()
        select_red_button = tk.Button(root, text="Select Red", command=self.__on_select_red)
        select_red_button.pack()

        tk.Button(root, text="Ok", command=self.__on_continue).pack()

    def __on_continue(self):
        super().stop()
        while self.root.winfo_children():
            self.root.winfo_children()[0].destroy()
        self.next_page_listener(self.camera_index ,self.green_selected, self.red_selected)

    def __on_mouse_click(self, event):
        x, y = event.x, event.y
        if self.actual_frame is None:
            return

        if self.red_selecting:
            self.red_selected = tuple(self.actual_frame[y, x])
            self.canvas.itemconfig(self.red_oval, fill=ColorCalibrationPage.__rgb_to_hex(self.red_selected))
        elif self.green_selecting:
            self.green_selected = tuple(self.actual_frame[y, x])
            self.canvas.itemconfig(self.green_oval, fill=ColorCalibrationPage.__rgb_to_hex(self.green_selected))

        self.green_selecting = False
        self.red_selecting = False

    def __on_select_red(self):
        self.green_selecting = False
        self.red_selecting = True

    def __on_select_green(self):
        self.red_selecting = False
        self.green_selecting = True