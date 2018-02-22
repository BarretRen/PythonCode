# 多线程小示例
import threading
import time

print("start the code")


def takeanother(name):
    time.sleep(5)
    print("I'm in a new thread, and wake up after 5 seconds.")
    print('hello ' + name)


threadObj = threading.Thread(target=takeanother, args=["Barret"])  # 创建Thread对象,传递函数需要的参数
# threadObj = threading.Thread(target=takeanother, kwargs={'name' : 'Barret'})  # 创建Thread对象,传递函数需要的参数
threadObj.start()  # 启动新创建的线程

print("end of the code")
