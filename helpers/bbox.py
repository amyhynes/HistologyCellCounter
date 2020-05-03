import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#A class which puts bounding boxes around image patches
class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def slice_to_bbox(slices):
    for s in slices:
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

def bbox_img_generator(image, patches):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    im = ax.imshow(image) 

    bboxes = slice_to_bbox(patches)
    i=1
    for bbox in bboxes:
        xwidth = bbox.x2 - bbox.x1
        ywidth = bbox.y2 - bbox.y1
        p = mpatches.Rectangle((bbox.x1, bbox.y1), xwidth, ywidth,
                              fc = 'none', ec = 'darkslateblue')
        ax.add_patch(p)
        ax.text(bbox.x2, bbox.y1, str(i), fontsize=8)
        i+=1

    plt.show()