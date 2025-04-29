import os
import time
#print ("24小时格式：" + str(int(time.strftime("%H")) *60 + int(time.strftime("%M"))) )

def WriteLog( message, flag=False):
    strMessage = '\n' + time.strftime('%Y-%m-%d %H:%M:%S')
    if flag:
        strMessage += ': %s' % message
    else:
        strMessage += ':\n%s' % message

    fileName = os.path.join(os.getcwd(), time.strftime('%Y-%m-%d') + '.txt')
    with open(fileName, 'a', encoding='utf-8') as f:
        f.write(strMessage)

#读取并修改文件内容
def ModfiyConfig():
    f = open('config.txt', 'r',encoding='utf-8')
    count = int(f.read())
    print(count)
    f.close()
    if count < int(time.strftime("%H")) * 60 + int(time.strftime("%M")) + 10:
        f = open('config.txt', 'w')
        f.write(str(count+1))
        f.close()




#存储在当前IDE文件夹下 ， message：消息内容 ，  filmname：文件名
def WriteHere( message, filmname):
    strMessage = '\n' + time.strftime('%Y-%m-%d %H:%M:%S')
    strMessage += ':\n%s' % message

    fileName = os.path.join(os.getcwd(), filmname + '.txt')
    print(fileName)
    with open(fileName, 'a', encoding='utf-8') as f:
        f.write(strMessage)



#存储在任意路径 ， message：消息内容 ， path：文件路径 ， filmname：文件名.WriteLog.WriteTxt("12345","D:/","log")
def WriteTxt( message, path, filmname):
    strMessage = '\n' + time.strftime('%Y-%m-%d %H:%M:%S')
    strMessage += ':\n%s' % message
    fileName = os.path.join(path, filmname + '.txt')
    with open(fileName, 'a', encoding='utf-8') as f:
        f.write(strMessage)


# 创建包含小时分钟的txt, 文件名filename, 写入msg, 保存路径path
def text_create_hm(filename, msg, path=''):
    # 路径为空则保存到当前路径
    if path == '':
        full_path = os.path.join(os.getcwd(), "{0}{1}{2}".format(filename, time.strftime('%Y%m%d%H%M'), '.txt'))
    else:
        full_path = '{0}{1}{2}{3}'.format(path, filename,time.strftime('%Y%m%d%H%M'), '.txt')
    file = open(full_path, 'a+')
    file.write('{0}> {1}'.format(time.strftime("%Y%m%d %H:%M:%S"), msg))
    file.write('\n')
    file.close()
    


# 创建txt, 文件名filename, 写入msg, 保存路径path
def text_create(filename, msg, path=''):
    # 路径为空则保存到当前路径
    if path == '':
        full_path = os.path.join(os.getcwd(), "{0}{1}{2}".format(filename, time.strftime('%Y%m%d'), '.txt'))
    else:
        full_path = '{0}{1}{2}{3}'.format(path, filename,time.strftime('%Y%m%d'), '.txt')
    file = open(full_path, 'a+')
    file.write('{0}> {1}'.format(time.strftime("%Y%m%d %H:%M:%S"), msg))
    file.write('\n')
    file.close()


#循环调用写日志
def run(interval):
    while True:
        try:
            # sleep for the remaining seconds of interval
            time_remaining = interval - time.time() % interval
            time.sleep(time_remaining)
            #ModfiyConfig()
        except Exception as e:
            print(e)



#主函数
if __name__ == '__main__':
    text_create("logs", "test", "")
    run(2)
    #ModfiyConfig()
    #WriteHere('aaa','bbb')
    #WriteTxt('aaa','d:\\','bbb')
