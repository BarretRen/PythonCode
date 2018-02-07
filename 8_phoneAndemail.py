# 查找剪切板中的电话号码和email
import pyperclip
import re

phoneRegex = re.compile(r'''(
(\d{3}|\(\d{3}\))?              #area code, example: 333,(333)
(\s|-|\.)?                      #separator
(\d{3})                         #first 3 digits
(\s|-|\.)                       #separator
(\d{4})                         # last 3 digits
(\s*(ext|x|ext.)\s*(\d{2,5}))?  #extension
)''', re.VERBOSE)



