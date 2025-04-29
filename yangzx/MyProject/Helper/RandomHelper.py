import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(2)

import tensorflow as tf
import random
import numpy as np
from keras import backend as K

def ResetRandomSeeds(seed=2):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 强制 TensorFlow 使用单线程。多线程是结果不可复现的一个潜在因素。
    # 更多详情，见: https://stackoverflow.com/questions/42022950/
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # `tf.set_random_seed()` 将会以 TensorFlow 为后端,在一个明确的初始状态下生成固定随机数字。
    # 更多详情，见: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.compat.v1.set_random_seed(1234)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
   
   
   
# 主函数
if __name__ == '__main__':
    os.environ['PYTHONHASHSEED']=str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)