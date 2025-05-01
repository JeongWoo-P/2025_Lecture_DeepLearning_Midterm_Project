#!/bin/bash

# 모델 목록
models=("efficientnet_b4" "efficientnet_b5")

for model in "${models[@]}"
do
  echo "🛠 Training with model: $model"

  # config.py 내 MODEL_NAME 변경
  sed -i "s/^MODEL_NAME\s*=.*/MODEL_NAME          = '$model'/" src/config.py

  # 학습 실행
  python train.py
done
