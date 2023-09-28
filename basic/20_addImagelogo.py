# 为指定目录下的图片添加logo并保存
import os
from PIL import Image

FIT_SIZE = 300  # 修改后图片的尺寸
LOGO_NAME = 'catlogo.png'
IMAGE_PATH = 'media'

logo = Image.open(os.path.join(IMAGE_PATH, LOGO_NAME))
logoWidth, logoHeight = logo.size

# Loop over all files in the working directory.
for file in os.listdir(IMAGE_PATH):
    if not (file.endswith(".png") or file.endswith(".jpg")) or file == LOGO_NAME:
        continue

    image = Image.open(os.path.join(IMAGE_PATH, file))
    width, height = image.size
    # Check if image needs to be resized.
    if logoWidth > FIT_SIZE and logoHeight > FIT_SIZE:
        # Calculate the new width and height to resize to.
        # 如果图像确实需要调整大小，就需要弄清楚它是太宽还是太高。
        # 如果 width大于height，则高度应该根据宽度同比例减小。这个比例是FIT_SIZE除以当前宽度的值
        if logoWidth > logoHeight:
            logoHeight = int((FIT_SIZE / logoWidth) * logoHeight)
            logoWidth = FIT_SIZE
        else:  # 反之，也进行同样的计算
            logoWidth = int((FIT_SIZE / logoHeight) * logoWidth)
            logoHeight = FIT_SIZE

    # Resize the image.
    logo = logo.resize((logoWidth, logoHeight))
    # Add the logo and save.
    image.paste(logo, (width - logoWidth, height - logoHeight), logo)
    image.save(os.path.join(IMAGE_PATH, 'addlogo.png'))
