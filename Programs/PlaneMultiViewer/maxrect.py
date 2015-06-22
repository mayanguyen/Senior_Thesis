# based on program by Keith O'Hara

from matplotlib import path 


def maxRect(img_w, img_h, quad, w, h):
    #img_w, img_h = 640, 480
    #quad = [[250, 50], [450, 20], [500, 450], [10, 400]]
    quadp = path.Path(quad) 
    #aspect = 3/4.0
    aspect = 1.0*h/w

    best_w = 0
    best_x = 0
    best_y = 0
    for x in range(img_w):
        for y in range(img_h):
            if quadp.contains_point([x, y])\
              and x<img_w - best_w\
              and y<img_h - (best_w * aspect):
                for xp in range(x+best_w, img_w):
                    w = xp - x
                    h = w * aspect
                    if quadp.contains_point([x + w, y]) and\
                       quadp.contains_point([x, y + h]) and\
                       quadp.contains_point([x + w, y + h]):
                        best_x = x
                        best_y = y
                        best_w = w
                        #print x, y, w, h
    return [[best_x, best_y], [best_x+best_w, best_y], [best_x+best_w, best_y+(best_w*aspect)], [best_x, best_y+(best_w*aspect)]]
                        