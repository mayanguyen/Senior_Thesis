# Keith O'Hara

from matplotlib import path 

SVG = True

if SVG:
    import svgwrite

width, height = 640, 480
quad = [[250, 50], [450, 20], [500, 450], [10, 400]]
quadp = path.Path(quad) 
aspect = 3/4.0

best_w = 0
for x in range(width):
    for y in range(height):
        if quadp.contains_point([x, y]):
            for xp in range(x+best_w, width):
                w = xp - x
                h = w * aspect
                if quadp.contains_point([x + w, y]) and\
                   quadp.contains_point([x, y + h]) and\
                   quadp.contains_point([x + w, y + h]):
                    best_w = w
                    print x, y, w, h               
                    if SVG:
                        dwg = svgwrite.Drawing('test.svg', profile='tiny')
                        dwg.add(dwg.polygon(quad, fill="none", 
                                            stroke=svgwrite.rgb(100, 10, 16, '%')))
                        dwg.add(dwg.rect((x, y), (w, h), 
                                         stroke=svgwrite.rgb(10, 10, 16, '%'), 
                                         fill = svgwrite.rgb(100, 100, 100, '%')))
                        dwg.save()
