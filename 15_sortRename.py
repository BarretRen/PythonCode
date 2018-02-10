# 将某一文件夹内带数字序号的文件排序，消除缺失的编号
import os
import re

path = 'D:\\CodeBase\\PythonNoGiveUp\\email'
regex = re.compile(r'^(.*?)(\d+)(.*?)$', re.VERBOSE)
numlist = []

for item in os.listdir(path):
    result = regex.search(item)
    if None != result:
        before = result.group(1)
        numlist.append(int(result.group(2)))
        after = result.group(3)

numlist.sort()
for i in range(1, len(numlist) + 1):
    if i != numlist[i - 1]:
        os.rename(path + '\\' + before + str(numlist[i - 1]) + after, path + '\\' + before + str(i) + after)
