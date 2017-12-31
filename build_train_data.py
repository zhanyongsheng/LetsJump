# 用来生成训练数据
# 在train_data文件夹会生成以数字顺序命名的图片
# time.npz文件用数组方式保存了每个图片对应的应该去按压的时间（毫秒）
# 可以使用两个手指同时按两个手机获取跳跃时间，其中一个手机写个app监听按压屏幕时间


import numpy as np
from PIL import Image
import os


def get_screenshot(num):
    os.system('adb shell screencap -p /sdcard/jump_temp.png')
    os.system('adb pull /sdcard/jump_temp.png .')
    im = Image.open(r"./jump_temp.png")
    w, h = im.size
    # 将图片压缩，并截取中间部分，截取后为100*100
    im = im.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    im = im.crop(region)
    # 转换为jpg
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, im)
    file_name = str(num) + ".jpg"
    bg.save(r"./train_data/" + file_name)


# touch_time_arr = []
# count = 0
touch_time_arr = np.load("./train_data/time.npz")["abc"].tolist()
print(touch_time_arr)
# count 当前到第几张图片
count = len(touch_time_arr)
print(count)

# 当输入为-1，重新生成图片；输入12345结束并保存
RESHOT = -1
FINISH = 12345

while True:

    print("shoting.....")
    get_screenshot(count)

    # 图片已生成，开始输入时间
    input_time = input(str(count) + " input touch time(ms): ")
    touchtime = int(input_time)

    # 如果输入-1，则重新生成图片
    if touchtime == RESHOT:
        continue

    # 如果输入12345，则保存并退出，此时多截了一张图，可以手动把最后一张删掉
    elif touchtime == FINISH:
        npr = np.array(touch_time_arr)
        np.savez("./train_data/time", abc=npr)
        break

    # 输入的是需要按压时间，保存到arr数组
    else:
        touchtime = touchtime / 1000
        touch_time_arr.append(touchtime)

    count = count + 1
