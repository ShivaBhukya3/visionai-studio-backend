import cv2
import numpy as np


def draw_text_with_background(img, text, pos, font_scale=0.5,
                               color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)
    return img


def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def draw_info_overlay(img, text_lines: list, position="top-left"):
    h, w = img.shape[:2]
    overlay = img.copy()
    pad = 10
    line_h = 22
    panel_h = len(text_lines) * line_h + pad * 2
    panel_w = max(len(t) for t in text_lines) * 8 + pad * 2

    if position == "top-left":
        x1, y1 = pad, pad
    elif position == "top-right":
        x1, y1 = w - panel_w - pad, pad
    else:
        x1, y1 = pad, pad

    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (x1 + pad, y1 + pad + (i + 1) * line_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1,
                    cv2.LINE_AA)
    return img
