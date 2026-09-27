try:
    import tkinter as tk
except Exception:
    tk = None

import cv2


def get_screen_size(default_width=1280, default_height=720):
    if tk is None:
        return default_width, default_height

    root = tk.Tk()
    root.withdraw()
    try:
        return root.winfo_screenwidth(), root.winfo_screenheight()
    finally:
        root.destroy()


def create_fullscreen_window(window_name, fullscreen=False):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    width, height = get_screen_size()
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, 0, 0)

    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)