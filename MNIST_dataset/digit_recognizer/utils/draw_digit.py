# Adapted from: ()
# https://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging
# https://stackoverflow.com/questions/52146562/python-tkinter-paint-how-to-paint-smoothly-and-save-images-with-a-different

from tkinter import *
import PIL
from PIL import Image, ImageDraw, ImageOps


def save():
    global image_number
    size = 28,28
    image1.thumbnail(size, Image.ANTIALIAS)
    image_save = PIL.ImageOps.invert(image1)
    filename = 'user_img.png'   # image_number increments by 1 at every save
    image_save.save(filename)


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=10)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=10)
    lastx, lasty = x, y


def open_canvas():
    global cv, draw, image1, lastx, lasty
    root = Tk()

    lastx, lasty = None, None

    cv = Canvas(root, width=300, height=300, bg='white')
    # --- PIL
    image1 = PIL.Image.new('LA', (300, 300), 'white')
    draw = ImageDraw.Draw(image1)

    cv.bind('<1>', activate_paint)
    cv.pack(expand=YES, fill=BOTH)

    btn_save = Button(text="save", command=save)
    btn_save.pack()

    root.mainloop()
