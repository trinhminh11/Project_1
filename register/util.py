import os
import pickle
import tkinter as tk
from tkinter import messagebox


def get_button(window, text, color, coords, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    button.place(x = coords[0], y = coords[1])

    return button


def get_img_label(window, coords, size):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    label.place(x = coords[0], y = coords[1], width = size[0], height=size[1])
    return label


def get_text_label(window, text, coords):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    label.place(x = coords[0], y = coords[1])
    return label


def get_entry_text(window, coords):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    inputtxt.place(x = coords[0], y = coords[1])
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


