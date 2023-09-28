# 多重剪切板
# py mcb.pyw save spam， 剪贴板中当前的内容用关键字 spam 保存
# py mcb.pyw spam， 重新spam代表的内容加载到剪贴板中
# py mcb.pyw list，将所有关键字的列表复制到剪贴板中
# py mcb.pyw delete spam, 删除spam代表的内容
# py mcb.pyw delete, 删除所有内容

import pyperclip
import shelve
import sys  # 用于读取命令行参数

shelf = shelve.open('mcb')

# Save clipboard content.
if len(sys.argv) == 3:
    if sys.argv[1] == 'save':
        shelf[sys.argv[2]] = pyperclip.paste()
    elif sys.argv[1] == 'delete':
        del shelf[sys.argv[2]]
elif len(sys.argv) == 2:
    if sys.argv[1] == 'list':
        pyperclip.copy(str(list(shelf.keys())))
    elif sys.argv[1] == 'delete':
        for key in shelf.keys():
            del shelf[key]
    elif sys.argv[1] in shelf.keys():
        pyperclip.copy(shelf[sys.argv[1]])

shelf.close()
