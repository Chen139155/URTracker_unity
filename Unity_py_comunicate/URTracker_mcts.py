import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import threading
import zmq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# ===== 状态类 =====
class ContinuousState:
    def __init__(self, e_c, e_m):
        self.e_c = max(0.0, min(1.0, e_c))  # clip to [0, 1]
        self.e_m = max(0.0, min(1.0, e_m))

    def to_tuple(self):
        return (round(self.e_c, 2), round(self.e_m, 2))

    def __repr__(self):
        return f"(e_c={self.e_c:.2f}, e_m={self.e_m:.2f})"

# ===== 状态转移模型（经验驱动） =====
class TransitionModel:
    def __init__(self):
        self.data = defaultdict(list)  # key: (e_c, e_m, action) -> [(e_c', e_m')]

    def update(self, state: ContinuousState, action, next_state: ContinuousState):
        key = (*state.to_tuple(), action)
        self.data[key].append(next_state.to_tuple())

    def sample(self, state: ContinuousState, action):
        key = (*state.to_tuple(), action)
        if key in self.data and len(self.data[key]) > 0:
            e_c, e_m = random.choice(self.data[key])
        else:
            # 未知动作，假设轻微扰动
            e_c = np.clip(state.e_c + np.random.normal(0, 0.05), 0, 1)
            e_m = np.clip(state.e_m + np.random.normal(0, 0.05), 0, 1)
        return ContinuousState(e_c, e_m)

# ===== 状态转移模型（高斯过程回归概率模型） =====
class EnhancedTransitionModel(TransitionModel):
    def __init__(self):
        super().__init__()
        # 初始化GP模型
        self.gp = {
            'e_c': GaussianProcessRegressor(kernel=RBF(length_scale=0.3)),
            'e_m': GaussianProcessRegressor(kernel=RBF(length_scale=0.3))
        }
        self.training_data = defaultdict(list)
        
    def update(self, state: ContinuousState, action, next_state: ContinuousState):
        super().update(state, action, next_state)
        
        # 收集训练数据
        self.training_data['X'].append([state.e_c, state.e_m, action])
        self.training_data['y_e_c'].append(next_state.e_c - state.e_c)
        self.training_data['y_e_m'].append(next_state.e_m - state.e_m)
        
        # 每50步重新训练模型
        if len(self.training_data['X']) % 50 == 0:
            X = np.array(self.training_data['X'])
            self.gp['e_c'].fit(X, self.training_data['y_e_c'])
            self.gp['e_m'].fit(X, self.training_data['y_e_m'])

    def sample(self, state: ContinuousState, action):
        key = (*state.to_tuple(), action)
        
        # 优先使用经验数据
        if key in self.data and len(self.data[key]) > 0:
            return super().sample(state, action)
            
        # GP预测（当经验数据不足时）
        X_pred = [[state.e_c, state.e_m, action]]
        delta_e_c = self.gp['e_c'].predict(X_pred)[0]
        delta_e_m = self.gp['e_m'].predict(X_pred)[0]
        
        # 添加不确定性探索
        noise_scale = 0.1 * (1 - np.exp(-len(self.data)/100))  # 随经验增长衰减
        delta_e_c += np.random.normal(0, noise_scale)
        delta_e_m += np.random.normal(0, noise_scale)
        
        return ContinuousState(
            np.clip(state.e_c + delta_e_c, 0, 1),
            np.clip(state.e_m + delta_e_m, 0, 1)
        )

# ===== MCTS 节点 =====
class MCTSNode:
    def __init__(self, state: ContinuousState, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> node
        self.visits = 0
        self.q_values = defaultdict(float)
        self.action_visits = defaultdict(int)

    def is_fully_expanded(self):
        return len(self.children) == 27

    def best_action(self, c=1.4):
        best_score = -float('inf')
        best_action = None
        for a in range(27):
            if self.action_visits[a] == 0:
                ucb = float('inf')
            else:
                exploit = self.q_values[a]
                explore = c * np.sqrt(np.log(self.visits + 1) / self.action_visits[a])
                ucb = exploit + explore
            if ucb > best_score:
                best_score = ucb
                best_action = a
        return best_action

# ===== MCTS 主类 =====
class MCTS:
    def __init__(self, transition_model, simulations=100, max_depth=5):
        self.P = transition_model
        self.simulations = simulations
        self.max_depth = max_depth

    def search(self, current_state: ContinuousState):
        self.root = MCTSNode(current_state)

        for _ in range(self.simulations):
            self.simulate(self.root, depth=0)

        best_action = max(self.root.q_values.items(), key=lambda x: x[1])[0]
        return best_action

    def simulate(self, node, depth):
        if depth >= self.max_depth:
            return self.reward(node.state)

        action = node.best_action()

        if action not in node.children:
            next_state = self.P.sample(node.state, action)
            child = MCTSNode(next_state, parent=node)
            node.children[action] = child
        else:
            child = node.children[action]

        reward = self.simulate(child, depth + 1)

        node.visits += 1
        node.action_visits[action] += 1
        node.q_values[action] += (reward - node.q_values[action]) / node.action_visits[action]
        return reward

    # ===== 奖励函数：e越小越好 ===== 
    def reward(self, state: ContinuousState):
        w_c, w_m = 0.6, 0.4
        return np.exp(- (w_c * state.e_c**2 + w_m * state.e_m**2))  # ∈ (0, 1]

class MCTSControlThread(threading.Thread):
    def __init__(self, sub_socket, cmd_q, pub_socket):
        super().__init__(daemon=True)
        self.sub_socket = sub_socket
        # self.result_q = result_q
        self.cmd_q = cmd_q
        self.pub_socket = pub_socket
        self.running = True
        
        # 初始化 MCTS 控制器
        self.transition_model = TransitionModel()
        self.mcts = MCTS(self.transition_model, simulations=200, max_depth=8)

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sub_socket, zmq.POLLIN)
        while self.running:
            try:
                events = dict(poller.poll(10))  # 10ms 超时
                if self.sub_socket in events:
                    result = self.sub_socket.recv_json()
                    # if result['status'] == 'success':
                    self.process_result(result)
            except Exception as e:
                print(f"MCTS 线程错误: {e}")

    def process_result(self, data):
        # 1. 构建当前状态
        current_state = ContinuousState(e_c=data['e_cog_norm'], 
                                      e_m=data['e_mot_norm'])
        
        # 2. MCTS 决策
        action = self.mcts.search(current_state)
        
        # 3. 解析动作为控制参数
        control_params = self.decode_action(action)
        
        # 4. 发送到 Unity 和机器人
        self.send_to_unity(control_params)
        self.send_to_robot(control_params)

    def decode_action(self, action):
        """将 0-26 的动作解码为三维参数 (基于你的 MCTS 设计)"""
        # 示例解码逻辑：三维参数每维3个等级 (0-2)
        x = (action // 9) % 3       # 0-2  难度
        y = (action // 3) % 3       # 0-2  反馈程度
        z = action % 3              # 0-2  辅助程度
        return {
            'diffculty': x,  # 0.5-1.0
            'feedback': y,     # 0.2-0.4
            'assistance': z        # 0-1.0
        }

    def send_to_unity(self, params):
        """通过 ZMQ 发送到 Unity 可视化"""
        message = {
            'type': 'mcts_params',
            'diffculty': params['diffculty'],
            'feedback': params['feedback'],
            'assistance': params['assistance']
        }
        self.pub_socket.send_json(message)

    def send_to_robot(self, params):
        """发送到 UR5 控制队列"""
        self.cmd_q.put('mode{}'.format(params['assistance']))

    def stop(self):
        self.running = False

# ===== 主循环示例 =====
if __name__ == "__main__":
    transition_model = TransitionModel()
    mcts = MCTS(transition_model, simulations=100, max_depth=10)

    # 初始状态（注意力异常高，运动正常）
    current_state = ContinuousState(e_c=0.8, e_m=0.2)
        # 初始化记录列表
    ec_values = []
    em_values = []

    for step in range(100):
        ec_values.append(current_state.e_c)
        em_values.append(current_state.e_m)     
        print(f"\nStep {step}, 当前状态: {current_state}")
        action = mcts.search(current_state)
        print(f"推荐动作: {action}")

        next_state = transition_model.sample(current_state, action)
        print(f"执行后状态: {next_state}")

        transition_model.update(current_state, action, next_state)
        current_state = next_state
    # 绘制趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(ec_values, label='e_c (Cognitive Error)', marker='o', markersize=3)
    plt.plot(em_values, label='e_m (Motor Error)', marker='s', markersize=3)
    plt.title('Error Dynamics over 100 Steps')
    plt.xlabel('Step')
    plt.ylabel('Error Value')
    plt.legend()
    plt.grid(True)
    plt.show()