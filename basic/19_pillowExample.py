# pillow模块操作图像示例
from PIL import Image

cat = Image.open('media/cat.png')  # 打开图像文件
# 打印图像属性
print(cat.size)  # 宽，高元组
print(cat.filename)
print(cat.format)
cat.save('media/cat.png')

im = Image.new('RGBA', (100, 100), 'black')  # 创建新图像
# im.save('media/new.png')

# 裁剪图像
cropcat = cat.crop((335, 345, 565, 560))
cropcat.save('media/cropcat.png')

# 复制粘贴图像
copycat = cat.copy()
copycat.paste(cropcat, (0, 0))
copycat.save('media/paste.png')

# 修改单个像素
for x in range(100):
    for y in range(50):
        im.putpixel((x,y), (210, 210, 210))

im.save('media/new.png')
