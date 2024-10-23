import numpy as np
import cv2


def center_place(image):
    res_y, res_x = np.shape(image)
    ax = np.linspace(0, res_x, res_x)
    ay = np.linspace(0, res_y, res_y)

    ax, ay = np.meshgrid(ax, ay)

    x_c = int(np.sum(ax * image) / np.sum(image))
    y_c = int(np.sum(ay * image) / np.sum(image))

    return x_c, y_c


def center_the_contour_all(shot):

    contours, hierarchy = cv2.findContours(shot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_x, c_y = center_place(shot)

    item = None
    for contour in contours:
        distance = cv2.pointPolygonTest(contour, (c_x, c_y), False)
        if distance >= 0:
            item = contour
    return (c_x, c_y), item, contours


def draw_centered_contour(shot):
    center, contour, contours = center_the_contour_all(shot)
    shot = cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(shot, [contour], -1, (0, 255, 0), 1)
    b, g, r = cv2.split(shot)

    c_x, c_y = center
    b[int(c_y)] = 0
    b[:, int(c_x)] = 0

    g[int(c_y)] = 255
    g[:, int(c_x)] = 255

    r[int(c_y)] = 0
    r[:, int(c_x)] = 0

    cv2.imshow('CONTOUR', cv2.merge([b, g, r]))

    return center, contour, contours


if __name__ == '__main__':


    img = cv2.imread(r'D:\PYTHON PROJECTS\2024\test\images\traps.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 10, 255, cv2.THRESH_TOZERO)


    c, _, _ = draw_centered_contour(img)
    print(c)
    cv2.waitKey(0)
