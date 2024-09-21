


def coorImg2Game(x_img, y_img):
    x_g = 0.5493 * x_img - 4141.1568
    y_g = -0.5488 * y_img + 8390.3183
    return x_g, y_g


if __name__ == '__main__':
    x_img, y_img= 3652, 15340
    print(coorImg2Game(x_img, y_img))

