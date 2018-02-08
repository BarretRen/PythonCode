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

emailRegex = re.compile(r'''(
[a-zA-Z0-9._%+-]+               #username
@
[a-zA-Z0-9.-]+                  #domain name
(\.[a-zA-Z]{2,4})               #dot something
)''', re.VERBOSE)

# find matches in clipboard text
text = str(pyperclip.paste())
matches = []
for group in phoneRegex.findall(text):
    phoneNum = '-'.join([group[1], group[3], group[5]])
    if group[8] != '':
        phoneNum += ' x' + group[8]
    matches.append(phoneNum)

for group in emailRegex.findall(text):
    matches.append(group[0])

# copy result to clipboard
if len(matches) > 0:
    pyperclip.copy('\n'.join(matches))
    print('Copied to clipboard: {0}'.format('\n'.join(matches)))
else:
    print('no phone number or email address')
