#!/bin/bash
# 自动执行tensorboard可视化的脚本
# 第一个参数即--logdir需要的参数
conda activate tensorboard
# if [ ! -n "$1" ];then
#   echo "please input the directory you want to visualize, with prefix visualize/gzsl/ given"
#   echo ""
# else
# 三层的情况
tensorboard --logdir=visualize/three_layer_gzsl/modified_multi_steplr3 --port 8123
# tensorboard --logdir=visualize/three_layer_gzsl/multi_steplr3_wd --port 8123
# 两层的情况
# tensorboard --logdir=visualize/gzsl/multi_steplr3 --port 8123
# fi

