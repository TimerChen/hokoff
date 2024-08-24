import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建一个空的 DataFrame 用于存放所有数据
all_data = pd.DataFrame(columns=['model_iter', 'win_rate', 'reward0', 'reward1'])

# 读取所有文件并合并数据
for i in range(50):  # 假设文件从 0 到 50
    file_path = "../offline_eval/offline_logs/mix15_1v1cql/win_rate_details/" + f"offline_win_rate_{i}.txt"
    if os.path.exists(file_path):
        print("open success")
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['model_iter', 'win_rate', 'reward0', 'reward1'])
        all_data = all_data.append(data, ignore_index=True)
        


# 去除 'win_rate'、'reward0' 和 'reward1' 列中的文本，只保留数值部分
all_data['win_rate'] = all_data['win_rate'].str.extract('(\d+\.\d+)', expand=False).astype(float)
all_data['reward0'] = all_data['reward0'].str.extract('(\d+\.\d+)', expand=False).astype(float)
all_data['reward1'] = all_data['reward1'].str.extract('(\d+\.\d+)', expand=False).astype(float)

# 按 'model_iter' 分组
grouped_data = all_data.groupby('model_iter')

# 计算平均值
mean_data = grouped_data.mean().reset_index()

# 绘制图表
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制平均 win_rate 图表
axs[0].plot(range(len(mean_data['model_iter'])), mean_data['win_rate'], marker='o', linestyle='-', color='b', label='Win Rate')
axs[0].set_title('Average Win Rate')
axs[0].set_xlabel('Model Iteration')
axs[0].set_ylabel('Win Rate')
axs[0].grid()
axs[0].legend()
axs[0].xaxis.set_tick_params(labelbottom=False)  # 隐藏 x 轴标签

# 绘制平均 reward0 图表
axs[1].plot(range(len(mean_data['model_iter'])), mean_data['reward0'], marker='o', linestyle='-', color='r', label='Reward0')
axs[1].set_title('Average Reward0')
axs[1].set_xlabel('Model Iteration')
axs[1].set_ylabel('Reward0')
axs[1].grid()
axs[1].legend()
axs[1].xaxis.set_tick_params(labelbottom=False)  # 隐藏 x 轴标签

# 绘制平均 reward1 图表
axs[2].plot(range(len(mean_data['model_iter'])), mean_data['reward1'], marker='o', linestyle='-', color='g', label='Reward1')
axs[2].set_title('Average Reward1')
axs[2].set_xlabel('Model Iteration')
axs[2].set_ylabel('Reward1')
axs[2].grid()
axs[2].legend()
axs[2].xaxis.set_tick_params(labelbottom=False)  # 隐藏 x 轴标签

# 调整布局
plt.tight_layout()

# 保存为图片
plt.savefig('average_results.png')

# 显示图表
plt.show()