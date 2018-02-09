# 读入文件，文本中出现 ADJECTIVE、 NOUN、 ADVERB 或 VERB的地方，替换为用户输入的文本
import re

regex = re.compile(r'(adjective|noun|verb)', re.I)
try:
    inFile = open('madlib.txt', 'r')
    text = inFile.read()
    inFile.close()
    print(text)
    for item in regex.findall(text):
        if item.lower() == 'adjective':
            adjective = input('Enter an adjective:\n')
            text = re.sub(r'(adjective)', adjective, text, 1, re.I)
        elif item.lower() == 'noun':
            noun = input('Enter a noun:\n')
            text = re.sub(r'(noun)', noun, text, 1, re.I)
        elif item.lower() == 'verb':
            verb = input('Enter a verb:\n')
            text = re.sub(r'(verb)', verb, text, 1, re.I)
    print(text)
except:
    print('invalid operation\n')
