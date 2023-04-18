#!/usr/bin/env python3

import math

import numpy as np
from PIL import Image, ImageDraw

# import matplotlib.pyplot as plt

from metrics import ROM, RUM


def sample_os():
    # single class sample

    # Define image size and background color
    bg_color = "black"

    width, height = 64, 64

    # Create a new image with white background
    gt = Image.new("L", (width, height), bg_color)

    # Create a drawing object
    gt_draw = ImageDraw.Draw(gt)

    # Define a list of shapes to draw
    shapes = [
        {"type": "rectangle", "coordinates": [(2, 2), (28, 60)]},
        {"type": "rectangle", "coordinates": [(32, 2), (60, 60)]}
        # {"type": "circle", "coordinates": [(110, 48), 10]},
        # {"type": "line", "coordinates": [(50, 50), (450, 450)]},
    ]

    # Draw each shape
    for shape in shapes:
        if shape["type"] == "rectangle":
            gt_draw.rectangle(shape["coordinates"], fill=1)
        elif shape["type"] == "circle":
            x, y = shape["coordinates"][0]
            r = shape["coordinates"][1]
            gt_draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        elif shape["type"] == "line":
            gt_draw.line(shape["coordinates"], fill="black", width=5)

    # Display the image
    # gt.show()

    pred = Image.new("L", (width, height), bg_color)
    pred_draw = ImageDraw.Draw(pred)
    shapes = [
        {"type": "rectangle", "coordinates": [(4, 4), (26, 26)]},
        {"type": "rectangle", "coordinates": [(4, 34), (26, 56)]},
    ]

    for shape in shapes:
        if shape["type"] == "rectangle":
            pred_draw.rectangle(shape["coordinates"], fill=1)
        elif shape["type"] == "circle":
            x, y = shape["coordinates"][0]
            r = shape["coordinates"][1]
            pred_draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        elif shape["type"] == "line":
            pred_draw.line(shape["coordinates"], fill="black", width=5)

    gt = np.asarray(gt)
    pred = np.asarray(pred)

    return gt, pred


def sample_us():
    # single class sample

    # Define image size and background color
    bg_color = "black"

    width, height = 64, 64

    # Create a new image with white background
    pred = Image.new("L", (width, height), bg_color)

    # Create a drawing object
    pred_draw = ImageDraw.Draw(pred)

    # Define a list of shapes to draw
    shapes = [
        {"type": "rectangle", "coordinates": [(2, 2), (28, 60)]},
        {"type": "rectangle", "coordinates": [(32, 2), (60, 60)]}
        # {"type": "circle", "coordinates": [(110, 48), 10]},
        # {"type": "line", "coordinates": [(50, 50), (450, 450)]},
    ]

    # Draw each shape
    for shape in shapes:
        if shape["type"] == "rectangle":
            pred_draw.rectangle(shape["coordinates"], fill=1)
        elif shape["type"] == "circle":
            x, y = shape["coordinates"][0]
            r = shape["coordinates"][1]
            pred_draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        elif shape["type"] == "line":
            pred_draw.line(shape["coordinates"], fill="black", width=5)

    # gt
    gt = Image.new("L", (width, height), bg_color)
    gt_draw = ImageDraw.Draw(gt)
    shapes = [
        {"type": "rectangle", "coordinates": [(4, 4), (26, 26)]},
        {"type": "rectangle", "coordinates": [(4, 34), (26, 56)]},
    ]

    for shape in shapes:
        if shape["type"] == "rectangle":
            gt_draw.rectangle(shape["coordinates"], fill=1)
        elif shape["type"] == "circle":
            x, y = shape["coordinates"][0]
            r = shape["coordinates"][1]
            gt_draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        elif shape["type"] == "line":
            gt_draw.line(shape["coordinates"], fill="black", width=5)

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    return gt, pred


def test_ROM():
    gt, pred = sample_os()
    rom = ROM(gt, pred)
    assert rom == math.tanh(0.5)


def test_RUM():
    gt, pred = sample_us()
    rum = RUM(gt, pred)
    assert rum == math.tanh(0.5)
