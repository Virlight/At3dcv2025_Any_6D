from PIL import Image
# image_base = "cartoon-headshot.jpg"
# image_name = image_base.split(".")[0]
# img = Image.open(f"data/{image_base}").convert("RGB")
# img.save(f"data/{image_name}-clean.jpg")


for i in range(1, 11):
    image_path = f"/media/haoliang/windows/linux_share/housecat6d/scene{i:02d}/rgb/000000.png"
    img = Image.open(image_path).convert("RGB")
    img.save(f"data/000000_scene{i:02d}.png")
