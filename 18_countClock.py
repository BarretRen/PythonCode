# 简单的倒计时程序
# 从 60 倒数。倒数至 0 时播放声音文件（alarm.wav）
import time
import subprocess  # 用于启动计算机上其他程序或脚本

timeLeft = 5
while timeLeft > 0:
    print(timeLeft)
    time.sleep(1)
    timeLeft = timeLeft - 1

# 60秒倒计时结束后，播放声音
subprocess.Popen(['start', 'media/alarm.wav'], shell=True)  # for windows, 使用默认程序打开wav文件
