# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('../img/catdog.jpg')
d2l.plt.imshow(img); # 加分号只显示图

# bbox是bounding box的缩写
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

def bbox_to_rect(bbox, color): # 画出边界框，本函数已保存在d2lzh_pytorch中⽅便以后使⽤
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, ⾼)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
plt.show()