import configparser

def Conf():
    # 配置文件路径
    cfgpath = "../config/config.ini"

    # 创建管理对象
    conf = configparser.ConfigParser()
    # 读取ini文件
    conf.read(cfgpath, encoding="utf-8")

    return conf
