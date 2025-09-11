from PIL import Image
import os

def convert_folder_webp_to_png(input_folder, output_folder):
    """
    批量将input_folder下的所有webp图片转换为png，并保存到output_folder
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.webp'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            # 输出文件名
            out_name = os.path.splitext(filename)[0] + '.png'
            out_path = os.path.join(output_folder, out_name)
            img.save(out_path, "PNG")
            print(f"Converted: {filename} -> {out_name}")

def convert_webp_to_png(input_file, output_file=None):
    """
    将单个webp图片转换为png
    input_file: 输入的webp文件名
    output_file: 输出的png文件名，如果为None，则自动替换扩展名
    """
    if output_file is None:
        import os
        output_file = os.path.splitext(input_file)[0] + '.png'
    img = Image.open(input_file)
    img.save(output_file, 'PNG')
    print(f"Converted: {input_file} -> {output_file}")

if __name__ == "__main__":
    # 批量转换文件夹内所有 .webp
    convert_folder_webp_to_png('data', 'data')

    # 转换单个文件
    # convert_webp_to_png('your_image.webp', 'your_image.png')

    # 转换单个文件, 自动命名
    # convert_webp_to_png('your_image.webp')