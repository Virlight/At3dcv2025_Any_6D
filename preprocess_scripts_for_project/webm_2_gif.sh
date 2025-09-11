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

# 遍历指定目录下所有 .webm 文件
shopt -s nullglob  # 防止没有匹配到文件时返回原字符串
for f in "$INPUT_DIR"/*.webm; do
    # 去掉路径和扩展名
    filename="$(basename "$f" .webm)"
    output="$INPUT_DIR/${filename}.gif"

    echo "Converting '$f' to '$output'..."
    ffmpeg -i "$f" -vf "fps=10,scale=480:-1:flags=lanczos" -c:v gif "$output" -y
done

echo "All conversions done."
