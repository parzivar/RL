import numpy as np
import matplotlib.pyplot as plt

# 设置迷宫
maze = np.zeros((5, 5))
maze[1, 1] = -1  # 障碍物
maze[1, 2] = -1  # 障碍物
maze[2, 0] = -1  # 障碍物
maze[3, 2] = -1  # 障碍物
# maze[3, 3] = -1  # 障碍物
maze[3, 4] = -1  # 障碍物
# 设置起始和目标位置
start_position = (0, 0)
goal_position = (4, 4)

# 初始化 Q 表
Q = np.zeros((5, 5, 4))  # 4 表示 4 个动作（上、下、左、右）

# 动作映射：0 - 上, 1 - 下, 2 - 左, 3 - 右
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 学习参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率


# 选择动作
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(4)  # 探索：随机选择动作
    else:
        return np.argmax(Q[state])  # 利用：选择 Q 值最大的动作


# 更新 Q 值
def update_q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error


# 环境交互
def step(state, action):
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

    # 检查是否越界或撞到障碍物
    if (next_state[0] < 0 or next_state[0] >= 5 or
            next_state[1] < 0 or next_state[1] >= 5 or
            maze[next_state] == -1):
        next_state = state
        reward = -10
    elif next_state == goal_position:
        reward = 100
    else:
        reward = -1

    return next_state, reward


# 训练过程
num_episodes = 1000
for episode in range(num_episodes):
    state = start_position
    while state != goal_position:
        action = choose_action(state)
        next_state, reward = step(state, action)
        update_q(state, action, reward, next_state)
        state = next_state

# 结果测试
state = start_position
path = [state]
while state != goal_position:
    action = np.argmax(Q[state])
    state, _ = step(state, action)
    path.append(state)


# 绘制迷宫和路径
def plot_maze_and_path(maze, path):
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap='gray')

    # 绘制路径
    path_x, path_y = zip(*path)
    ax.plot(path_y, path_x, marker='o', color='r', linewidth=2, markersize=5)

    # 标记起点和终点
    ax.text(start_position[1], start_position[0], 'S', color='blue', ha='center', va='center', fontsize=12)
    ax.text(goal_position[1], goal_position[0], 'G', color='green', ha='center', va='center', fontsize=12)

    plt.title('Robot Path in the Maze')
    plt.show()


plot_maze_and_path(maze, path)