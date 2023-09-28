# zipfile模块处理zip格式压缩文件
import zipfile
import os

zip = zipfile.ZipFile('d:\\Desktop\Desktop.zip')
# 打印所有文件
print(zip.namelist())
# 获取某个文件对应的ZipInfo对象
info = zip.getinfo('Cas T&D.xlsx')
# 打印文件大小
print(info.file_size)
# 解压文件
zip.extractall('d:\\Desktop')
zip.close()

# 创建一个压缩文件
zip = zipfile.ZipFile('myzip.zip', 'w')
os.chdir('d:\\Desktop\CDE')
for item in os.listdir('.'):
    zip.write(item)

zip.close()
