from PIL import Image
import numpy as np
import os

def rgb_or_rgba_to_rgba_white_to_transparent(img: Image.Image) -> Image.Image:
    """处理 RGB 或 RGBA 图像，将白色背景变透明，并保留原透明区域"""
    img = img.convert("RGBA")  # 无论如何转为 RGBA
    np_img = np.array(img)     # (H, W, 4)

    rgb = np_img[:, :, :3]     # 提取 RGB 通道
    alpha = np_img[:, :, 3]    # 提取原始 alpha 通道

    # 判断哪些是“接近白色”的像素
    white_bg = np.all(rgb >= 255, axis=-1)

    # 将白色像素的 alpha 设置为 0，其余保留原 alpha
    alpha[white_bg] = 0

    # 合并回 RGBA
    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, mode='RGBA')

if __name__ == "__main__":

    input_dir = "data"
    output_dir = "data_rgba"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path)
            rgba_img = rgb_or_rgba_to_rgba_white_to_transparent(img)
            
            out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".png")
            rgba_img.save(out_path)
            print(f"Saved: {out_path}")