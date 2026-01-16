#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入必要的库
import os
import numpy as np
import pandas as pd
import json
import random
import math
from collections import defaultdict, namedtuple
import time

# 设置工作目录
WORK_DIR = r"d:\Unity\projecct\URTracker_unity\URTracker_unity\Unity_py_comunicate"
os.chdir(WORK_DIR)

# 定义状态和动作
STATE_LABELS = {
    0: 'Normal',
    1: 'Motor Abnormal',
    2: 'Attention Abnormal',
    3: 'Mixed Abnormal'
}

DifficultyLevels = ['low', 'medium', 'high']
FeedbackLevels = ['high', 'medium', 'low']
AssistanceLevels = ['high', 'medium', 'low']

Action = namedtuple("Action", ["difficulty", "feedback", "assistance"])

ACTIONS = []
for d in DifficultyLevels:
    for f in FeedbackLevels:
        for a in AssistanceLevels:
            ACTIONS.append(Action(d, f, a))

# 奖励函数
def compute_reward(state, action):
    state_score = {
        0: 4.0,
        1: 0.5,
        2: 0.5,
        3: 0.0
    }[state]
    
    difficulty_score = {'low': 0.0, 'medium': 0.5, 'high': 1.0}[action.difficulty]
    feedback_score   = {'high': 0.0, 'medium': 0.5, 'low': 1.0}[action.feedback]
    assistance_score = {'high': 0.0, 'medium': 0.5, 'low': 1.0}[action.assistance]

    challenge_score = (difficulty_score + feedback_score + assistance_score) / 3.0
    return 0.75 * state_score + 0.25 * challenge_score

# 状态转移模型
class TransitionModel:
    def __init__(self):
        self.probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def update(self, state, action, next_state):
        key = (state, action)
        if key not in self.counts:
            self.counts[key] = defaultdict(int)
        self.counts[key][next_state] += 1
        total = sum(self.counts[key].values())
        for ns in self.counts[key]:
            self.probabilities[key][ns] = self.counts[key][ns] / total

    def sample_next_state(self, state, action):
        key = (state, action)
        probs = self.probabilities[key]
        if not probs:
            return random.choice(list(STATE_LABELS.keys()))
        states, weights = zip(*probs.items())
        return random.choices(states, weights=weights)[0]

    def load(self, filepath):
        if not os.path.exists(filepath):
            print(f"{filepath} 不存在，使用空模型")
            return
        with open(filepath, 'r') as f:
            loaded_counts = json.load(f)
        for key_str, next_states in loaded_counts.items():
            state_str, d, fdbk, assist = key_str.split('|')
            state = int(state_str)
            action = Action(d, fdbk, assist)
            key = (state, action)
            self.counts[key] = defaultdict(int, {int(k): v for k, v in next_states.items()})
            total = sum(self.counts[key].values())
            for ns in self.counts[key]:
                self.probabilities[key][ns] = self.counts[key][ns] / total
        print(f"TransitionModel 已从 {filepath} 加载")

# MCTS节点与主类
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(ACTIONS)

    def best_child(self, c_param=1.0):
        choices = [
            (child, (child.value / (child.visits + 1e-5)) +
             c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-5)))
            for child in self.children
        ]
        return max(choices, key=lambda x: x[1])[0]

class MCTS:
    def __init__(self, transition_model, iterations=100):
        self.transition_model = transition_model
        self.iterations = iterations

    def search(self, initial_state):
        root = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        return root.best_child(c_param=0)

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        if not node.is_fully_expanded():
            return self.expand(node)
        return node

    def expand(self, node):
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in ACTIONS if a not in tried_actions]
        action = random.choice(untried_actions)
        next_state = self.transition_model.sample_next_state(node.state, action)
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        current_state = node.state
        total_reward = 0
        depth = 0
        max_depth = 5
        
        for _ in range(max_depth):
            action = random.choice(ACTIONS)
            next_state = self.transition_model.sample_next_state(current_state, action)
            reward = compute_reward(next_state, action)
            total_reward += reward * (0.9 ** depth)
            current_state = next_state
            depth += 1
            
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

# 导纳参数获取函数
def get_admittance_params_by_assistance_level(assistance_level):
    if assistance_level == 'high':
        return {'K': np.array([400, 400, 400]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}
    elif assistance_level == 'medium':
        return {'K': np.array([250, 250, 250]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}
    elif assistance_level == 'low':
        return {'K': np.array([5, 5, 5]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}
    else:
        return {'K': np.array([250, 250, 250]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}

# 参数插值函数
def interpolate_params(start_params, end_params, progress):
    interpolated = {}
    for key in ['K', 'D', 'M']:
        interpolated[key] = start_params[key] + progress * (end_params[key] - start_params[key])
    return interpolated

# 计算状态标志
def calculate_state_flag(e_c, e_m, tau_cog=2.3, tau_mot=1.5):
    if e_c < tau_cog and e_m < tau_mot:
        return 0  # Normal
    elif e_c < tau_cog and e_m >= tau_mot:
        return 1  # Motor Abnormal
    elif e_c >= tau_cog and e_m < tau_mot:
        return 2  # Attention Abnormal
    else:
        return 3  # Mixed Abnormal

# 处理单个数据文件
def process_data_file(input_file_path, output_file_path):
    # 加载数据
    print(f"正在处理文件: {input_file_path}")
    df = pd.read_excel(input_file_path)
    
    # 初始化转移模型和MCTS
    transition_model = TransitionModel()
    model_file = "model.json"
    if os.path.exists(model_file):
        transition_model.load(model_file)
    mcts = MCTS(transition_model=transition_model, iterations=100)
    
    # 初始化变量
    last_state = None
    last_action = None
    current_K = np.array([250, 250, 250])
    current_D = np.array([80, 80, 80])
    current_M = np.array([40, 40, 40])
    last_mcts_assistance = 'medium'
    
    # 添加时间间隔控制变量
    last_mcts_time = None  # 记录上一次MCTS决策的时间
    time_interval = 3.0    # MCTS决策时间间隔（秒）
    sample_interval = 0.02 # 数据采样间隔（秒），基于50Hz的采集频率
    
    # 重新处理每一行数据
    for index, row in df.iterrows():
        # 获取e_c和e_m
        e_c = row['e_c']
        e_m = row['e_m']
        
        # 计算state_flag
        state_flag = calculate_state_flag(e_c, e_m)
        df.at[index, 'state_flag'] = state_flag
        
        # 计算当前时间戳（假设第一行数据的时间为0）
        current_time = index * sample_interval
        
        # 执行MCTS搜索（仅当时间间隔达到3秒时）
        if last_state is not None and last_action is not None:
            # 更新转移模型
            transition_model.update(last_state, last_action, state_flag)
        
        # 仅当时间间隔达到3秒时执行MCTS决策
        if last_mcts_time is None or (current_time - last_mcts_time) >= time_interval:
            # 执行MCTS搜索
            best_action_node = mcts.search(state_flag)
            best_action = best_action_node.action
            
            # 更新last_mcts_time
            last_mcts_time = current_time
            
            # 更新last_state和last_action
            last_state = state_flag
            last_action = best_action
            
            # 更新导纳参数
            if best_action.assistance != last_mcts_assistance:
                # 渐变切换参数
                start_params = get_admittance_params_by_assistance_level(last_mcts_assistance)
                target_params = get_admittance_params_by_assistance_level(best_action.assistance)
                
                # 为了简化，这里直接使用目标参数，不进行渐变
                current_K = target_params['K']
                current_D = target_params['D']
                current_M = target_params['M']
                
                last_mcts_assistance = best_action.assistance
        
        # 更新MCTS参数（使用上一次的决策结果）
        if last_action is not None:
            df.at[index, 'mcts_difficulty'] = last_action.difficulty
            df.at[index, 'mcts_feedback'] = last_action.feedback
            df.at[index, 'mcts_assistance'] = last_action.assistance
        
        # 更新参数列
        df.at[index, 'K_0'] = float(current_K[0])
        df.at[index, 'K_1'] = float(current_K[1])
        df.at[index, 'K_2'] = float(current_K[2])
        df.at[index, 'D_0'] = float(current_D[0])
        df.at[index, 'D_1'] = float(current_D[1])
        df.at[index, 'D_2'] = float(current_D[2])
        df.at[index, 'M_0'] = float(current_M[0])
        df.at[index, 'M_1'] = float(current_M[1])
        df.at[index, 'M_2'] = float(current_M[2])
    
    # 保存处理后的数据
    df.to_excel(output_file_path, index=False)
    print(f"处理完成，结果保存到: {output_file_path}")

# 批量处理数据文件
def batch_process_data(input_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有Excel文件
    excel_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    
    # 处理每个文件
    for file in excel_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"processed_{file}")
        process_data_file(input_path, output_path)

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"D:\Unity\projecct\URTracker_unity\URTracker_unity\Unity_py_comunicate\Reprocess\raw"
    output_directory = r"D:\Unity\projecct\URTracker_unity\URTracker_unity\Unity_py_comunicate\Reprocess\output"
    
    # 执行批量处理
    batch_process_data(input_directory, output_directory)