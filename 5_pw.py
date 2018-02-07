#! python3
# pw.py
import sys
import pyperclip

PASSWORDS = {
    "email": "124",
    "blog": "556",
    "luggage": "778"
}

if len(sys.argv) < 2:
    print('Usage python pw.py [account]')
    sys.exit()

account = sys.argv[1]

if account in PASSWORDS:
    pyperclip.copy(PASSWORDS[account])
    print('password for' + account + 'copid to clipboard')
else:
    print('there si no account named' + account)
