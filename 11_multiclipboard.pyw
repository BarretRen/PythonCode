# 多重剪切板
# 当运行 py mcb.pyw save spam， 剪贴板中当前的内容就用关键字 spam 保存
# 通过运行 py mcb.pyw spam， 这段文本稍后将重新加载到剪贴板中
# 如果忘记了都有哪些关键字， 可以运行 py mcb.pyw list，将所有关键字的列表复制到剪贴板中

import pyperclip
import shelve
import sys  # 用于读取命令行参数

shelf = shelve.open('mcb')

# Save clipboard content.
if len(sys.argv) == 3 and sys.argv[1] =='save':
    shelf[sys.argv[2]] = pyperclip.paste()
elif len(sys.argv) == 2:
    if sys.argv[1] == 'list':
        pyperclip.copy(str(list(shelf.keys())))
    elif sys.argv[1] in shelf.keys():
        pyperclip.copy(shelf[sys.argv[1]])


shelf.close()
