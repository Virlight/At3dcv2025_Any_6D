#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

# 输入路径
INPUT_DIR="$1"

# 检查路径是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist."
    exit 1
fi

# 遍历指定目录下所有 .mp4 文件
for f in "$INPUT_DIR"/*.mp4; do
    # 如果没有匹配到文件，glob 会返回原字符串
    if [ ! -e "$f" ]; then
        echo "No .mp4 files found in '$INPUT_DIR'."
        exit 0
    fi

    # 去掉路径和扩展名
    filename="$(basename "$f" .mp4)"
    output="$INPUT_DIR/${filename}.gif"

    echo "Converting '$f' to '$output'..."
    ffmpeg -i "$f" -vf "fps=10,scale=480:-1:flags=lanczos" -c:v gif "$output" -y
done

echo "All conversions done."