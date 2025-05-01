import matplotlib.pyplot as plt

# 모델, FLOPs (Million), Accuracy
models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
          'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']
flops = [64.445, 95.343, 110.332, 160.956, 243.842, 394.679]
accuracy = [0.4196, 0.4116, 0.4172, 0.4142, 0.4523, 0.4155]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(flops, accuracy, marker='o', linestyle='-', color='orange')

# 모델명 라벨 각 점에 붙이기
for i, name in enumerate(models):
    plt.text(flops[i] + 5, accuracy[i], name, fontsize=9)

# 라벨 및 제목
plt.title('Size-Accuracy Trade-off (EfficientNet Family on TinyImageNet)')
plt.xlabel('FLOPs (Million)')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.tight_layout()

# 그래프 출력
plt.show()

# 그래프 저장
plt.savefig('size_accuracy_trade_off.png', dpi=300, bbox_inches='tight')