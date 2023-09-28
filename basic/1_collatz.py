def collatz(number):
    if number % 2 == 0:
        return number // 2
    else:
        return 3 * number + 1


st = input("输入一个整数:")
try:
    num = int(st)
    print(num)
    num = collatz(num)
    while 1 != num:
        print(num)
        num = collatz(num)

    print(num)
except ValueError:
    print("必须输入整数")
