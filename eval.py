import mnist
import tensorflow as tf
import os,sys
for i in range(1, 6):
    steps = 1000 * i
    if i == 5:
        steps = 4999
    with open('log/fully_connected_feed/checkpoint', 'w') as f:
        formatstr = 'model_checkpoint_path: "/tmp/material_train/model.ckpt-%d"\n'
        f.write(formatstr % steps)
    os.system("python DeepFullyConnected_eval.py")
