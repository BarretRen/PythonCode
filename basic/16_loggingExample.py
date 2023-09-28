# logging 模块使用
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s -%(message)s')
logging.debug('Start the program')


def factorial(n):
    logging.debug('Enter function factorail with input {0}'.format(n))
    total = 1
    for i in range(n + 1):
        total *= i
        logging.debug('i is {0}, total is {1}'.format(i, total))

    logging.debug('Exit function factorial')
    return total


print(factorial(5))
logging.debug('End the program')
