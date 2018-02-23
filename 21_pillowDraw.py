# pillow 绘图和文本
from PIL import Image, ImageDraw

im = Image.new('RGBA', (200, 200), 'white')
draw = ImageDraw.Draw(im)
draw.line([(0, 0), (199, 0), (199, 199), (0, 199), (0, 0)], fill='black')
draw.rectangle((20, 30, 60, 60), fill='blue')
draw.ellipse((120, 30, 160, 60), fill='red')  # 椭圆
draw.polygon(((57, 87), (79, 62), (94, 85), (120, 90), (103, 113)),fill='brown')  # 多边形

draw.text((20, 150), 'Hello', fill='purple')

im.save('media/drawing.png')