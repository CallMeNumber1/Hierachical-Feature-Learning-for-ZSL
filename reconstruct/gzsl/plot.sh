#!/bin/bash
# 自动执行tensorboard可视化的脚本
# 第一个参数即--logdir需要的参数
conda activate tensorboard
if [ ! -n "$1" ];then
  echo "please input the directory you want to visualize, with prefix visualize/gzsl/ given"
  echo ""
else
  tensorboard --logdir=visualize/gzsl/$1 --port 8123
fi

