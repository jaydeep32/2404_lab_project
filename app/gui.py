# ---------------------------------------------
import tkinter as tk
from tkinter import *

import numpy as np
# import win32gui
from PIL import ImageGrab
from app import app
from keras.models import load_model


def predict_digit(img):
    class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
    model = load_model(app.config["STATIC_PATH"] / 'mnist.h5')
    # model = load_model(app.config["STATIC_PATH"] / 'full_model.h5')

    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = np.argmax(model.predict(img))
    return class_mapping[res]


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    # def classify_handwriting(self):
    #     HWND = self.canvas.winfo_id()  # get the handle of the canvas
    #     rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
    #     im = ImageGrab.grab(rect)
    #
    #     digit, acc = predict_digit(im)
    #     self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def classify_handwriting(self):
        x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
        im = ImageGrab.grab((x, y, x + self.canvas.winfo_width(), y + self.canvas.winfo_height()))

        digit = predict_digit(im)
        self.label.configure(text=str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


def start_gui():
    app = App()
    mainloop()


if __name__ == "__main__":
    start_gui()
