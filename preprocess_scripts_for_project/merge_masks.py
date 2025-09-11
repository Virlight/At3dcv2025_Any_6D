import numpy as np
from PIL import Image

files = [
    'data/000000_scene2_rectangular_box.png',
    'data/000000_scene2_controller.png',
    'data/000000_scene2_fork.png',
    'data/000000_scene2_can.png',
    'data/000000_scene2_cup.png',
    'data/000000_scene2_tape.png',
    'data/000000_scene2_bottle.png',
    # 你可以继续加更多文件
]

# 读入所有图片
images = [np.array(Image.open(f).convert('RGB')) for f in files]

# 检查所有图片尺寸一致
shapes = [img.shape for img in images]
assert all(s == shapes[0] for s in shapes), "所有输入图片尺寸需一致！"

# 初始化合成结果为全白
composite = np.ones_like(images[0]) * 255

# 前景判据：不是全白就是前景
for img in images:
    is_fg = np.any(img < 250, axis=-1)
    composite[is_fg] = img[is_fg]

# 保存结果
Image.fromarray(composite).save('data/combined.png')
