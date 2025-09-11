import os
from PIL import Image

def remove_alpha_channel(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.tiff', '.tif', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)
            if img.mode == 'RGBA':
                # 去掉alpha通道，转换成RGB
                rgb_img = img.convert('RGB')
                output_path = os.path.join(output_folder, filename)
                rgb_img.save(output_path)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Skipped (not RGBA): {filename}")

if __name__ == "__main__":
    input_folder = "data_eval_rembg_flip"    # 替换成你的输入文件夹路径
    output_folder = "data_eval_rembg_flip_rgb"  # 替换成你的输出文件夹路径
    remove_alpha_channel(input_folder, output_folder)
