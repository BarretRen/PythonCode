#! python3
# copy from clipboard,add * in each line and then send to clipboard

import pyperclip

#  text = pyperclip.paste()
text = 'Lists of animals\nLists of aquarium life\nLists of biologists by authorabbreviation\nLists of cultivars'
# modify text
lines = text.split('\n')
text = ''
for line in lines:
    text += '* ' + line + '\n'

#  pyperclip.copy(text)
print(text)
