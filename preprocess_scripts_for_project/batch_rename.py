import os

def batch_rename(folder_path, id):
    files = os.listdir(folder_path)
    for filename in files:
        old_path = os.path.join(folder_path, filename)
        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{id}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    # folder = input("请输入文件夹路径: ")
    # id = input("请输入ID: ")
    for i in range(0, 11):
        id = i
        folder = f"/home/haoliang/workspace/3dcv-practice/InstantMesh/output_eval_gt_focus2_{id}/instant-mesh-large/meshes"
        batch_rename(folder, id)