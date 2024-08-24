import matplotlib.pyplot as plt

# 新的数据（根据您的输出更新）
average_win_rates = {
    0: (0.11, -4.0),
    1: (0.07, -4.15),
    2: (0.12, -3.52),
    3: (0.12, -3.83),
    4: (0.19, -4.37),
    5: (0.11, -3.97),
    6: (0.24, -4.1),
    7: (0.17, -3.73),
    8: (0.14, -4.3),
    9: (0.22, -3.67),
    10: (0.67, -4.48),
    11: (0.75, -5.12),
    12: (0.72, -4.18),
    13: (0.81, -4.28),
    14: (0.83, -5.42),
    15: (0.93, -3.47),
    16: (0.85, -5.05),
    17: (0.95, -3.8),
    18: (0.81, -4.54),
    19: (0.88, -4.43),
}

# 提取胜率和奖励数据
win_rates = [avg[0] for avg in average_win_rates.values()]
rewards = [avg[1] for avg in average_win_rates.values()]

# 将数据分为两组
group_1_win_rates = win_rates[0:10]
group_1_rewards = rewards[0:10]
group_2_win_rates = win_rates[10:20]
group_2_rewards = rewards[10:20]

# 训练步骤
train_steps_group_1 = [i * 5 for i in range(1, 11)]
train_steps_group_2 = [i * 5 for i in range(1, 11)]  # 从 10 开始

# 创建图形
plt.figure(figsize=(12, 12))

# 绘制第一组
plt.subplot(2, 2, 1)  # 2行2列的第1个子图
plt.plot(train_steps_group_1, group_1_win_rates, marker='o', linestyle='-', color='b', label='Win Rate')
plt.title('Win Rate (vs level-5)')
plt.xlabel('Train Steps(*10000)')
plt.ylabel('Win Rate')
plt.xticks(train_steps_group_1)  # 设置 x 轴刻度
plt.ylim(0, 1)  # 假设胜率在 0 到 1 之间
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)  # 2行2列的第2个子图
plt.plot(train_steps_group_1, group_1_rewards, marker='o', linestyle='-', color='r', label='Reward')
plt.title('Reward (vs level-5)')
plt.xlabel('Train Steps(*10000)')
plt.ylabel('Reward')
plt.xticks(train_steps_group_1)  # 设置 x 轴刻度
plt.grid()
plt.legend()

# 绘制第二组
plt.subplot(2, 2, 3)  # 2行2列的第3个子图
plt.plot(train_steps_group_2, group_2_win_rates, marker='o', linestyle='-', color='b', label='Win Rate')
plt.title('Win Rate (vs level-5)')
plt.xlabel('Train Steps(*10000)')
plt.ylabel('Win Rate')
plt.xticks(train_steps_group_2)  # 设置 x 轴刻度
plt.ylim(0, 1)  # 假设胜率在 0 到 1 之间
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)  # 2行2列的第4个子图
plt.plot(train_steps_group_2, group_2_rewards, marker='o', linestyle='-', color='r', label='Reward')
plt.title('Reward (vs level-1)')
plt.xlabel('Train Steps(*10000)')
plt.ylabel('Reward')
plt.xticks(train_steps_group_2)  # 设置 x 轴刻度
plt.grid()
plt.legend()

# 调整布局
plt.tight_layout()

# 导出为图片
plt.savefig('win_rate_and_rewards_vs_steps_groups.png', dpi=300)  # 保存为 PNG 格式，分辨率为 300 DPI

# 显示图形
plt.show()
