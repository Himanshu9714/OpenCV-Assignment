#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
from rembg import remove


def change_matrix(input_mat, stroke_size):
    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)
    return mat


def cv2pil(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(cv_img.astype("uint8"))
    return pil_img


def stroke(origin_image, threshold, stroke_size, colors):
    img = np.array(origin_image)
    h, w, _ = img.shape
    padding = stroke_size + 50
    alpha = img[:, :, 3]
    rgb_img = img[:, :, 0:3]
    bigger_img = cv2.copyMakeBorder(
        rgb_img,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )
    alpha = cv2.copyMakeBorder(
        alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )
    bigger_img = cv2.merge((bigger_img, alpha))
    h, w, _ = bigger_img.shape

    _, alpha_without_shadow = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(
        alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3
    )  # dist l1 : L1 , dist l2 : l2
    stroked = change_matrix(dist, stroke_size)
    stroke_alpha = (stroked * 255).astype(np.uint8)

    stroke_b = np.full((h, w), colors[0][2], np.uint8)
    stroke_g = np.full((h, w), colors[0][1], np.uint8)
    stroke_r = np.full((h, w), colors[0][0], np.uint8)

    stroke = cv2.merge((stroke_b, stroke_g, stroke_r, stroke_alpha))
    stroke = cv2pil(stroke)
    bigger_img = cv2pil(bigger_img)
    result = Image.alpha_composite(stroke, bigger_img)
    return result


def read_image(filepath):
    img = cv2.imread(filepath)
    return img


def scale_and_resize_image(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img


def crop_image(img):
    # Select ROI
    r = cv2.selectROI("select the area", img)

    # Crop image
    cropped_image = img[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]

    return cropped_image, (int(r[1]), int(r[1] + r[3]), int(r[0]), int(r[0] + r[2]))


def remove_bg(img):
    img = remove(img)
    return img


if __name__ == "__main__":

    # Read image
    filepath = r"TEST_IMAGES\2.jpg"
    img = read_image(filepath=filepath)

    # Resize image
    img = scale_and_resize_image(img)

    # Crop image
    cropped_img, coordinates = crop_image(img)

    # Remove cropped image background
    remove_img = remove_bg(cropped_img)

    # Draw outline to the removed background image
    output = stroke(remove_img, threshold=0, stroke_size=10, colors=((0, 255, 0),))

    # Show image with outlines
    output.show()

    # TODO: Overlay the outlined image to original image
    # TODO: Allow user to select again from the new image
    # TODO: Allow user to quit the process
