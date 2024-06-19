from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# 打开原始图片
input_path = '../satellite/7.png'
img = Image.open(input_path)

# 获取原始图片尺寸
original_width, original_height = img.size

# 目标尺寸
target_width = 24576
target_height = 24576

# 创建一个新的透明图像
new_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

# 计算粘贴位置，使得原始图片居中
x_offset = 0
y_offset = 0

# 将原始图片粘贴到新图像上
new_img.paste(img, (x_offset, y_offset))

# 保存结果图片
output_path = '../satellite/GTAV_satellite_square.png'
new_img.save(output_path)

print(f"New image saved at {output_path}")