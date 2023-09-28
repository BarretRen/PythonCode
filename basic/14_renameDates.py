# 文件批量重命名，将MM-DD-YYYY文件重命名为DD-MM-YYYY
import os
import re

# 1. 定义匹配MM-DD-YYYY的正则表达式
regex = re.compile(r'''
^(.*?)          # 所有date之前的文本
((0|1)?\d)-     # 月份
((0|1|2|3)?\d)- # 天数
((19|20)\d\d)   # 年
(.*?)$          # date之后的文本
''', re.VERBOSE)

# 2.遍历指定目录所有文件
for item in os.listdir('email'):
    # 3.正则查找，跳过无日期文件
    result = regex.search(item)
    if None == result:
        continue
    # 3.获取文件名各部分
    before = result.group(1)
    month = result.group(2)
    day = result.group(4)
    year = result.group(6)
    after = result.group(8)
    # 4. 重组文件名
    filename = before + day + '-' + month + '-' + year + after
    # 5. 获取绝对路径
    dir = os.path.abspath('email')
    oldname = os.path.join(dir, item)
    newname = os.path.join(dir, filename)
    print('new file name: ' + newname)
    # rename
    os.rename(oldname, newname)
