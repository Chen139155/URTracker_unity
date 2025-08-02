import numpy as np
import random
from collections import defaultdict

# 状态枚举
STATE_NORMAL = 0
STATE_ATTENTION_ABNORMAL = 1
STATE_MOTOR_ABNORMAL = 2
STATE_MIXED_ABNORMAL = 3
ALL_STATES = [0, 1, 2, 3]

# 动作空间（27个组合）：0~26
ALL_ACTIONS = list(range(27))

# 状态转移概率库
class TransitionProbability:
    def __init__(self):
        self.counts = np.ones((4, 27, 4))  # 初始值为1，Dirichlet平滑
        self.update_probs()

    def update(self, s, a, s_next):
        self.counts[s][a][s_next] += 1
        self.update_probs()

    def update_probs(self):
        self.probs = self.counts / self.counts.sum(axis=2, keepdims=True)

    def sample(self, s, a):
        return np.random.choice(ALL_STATES, p=self.probs[s][a])

    def get(self, s, a):
        return self.probs[s][a]

# MCTS树节点
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action: MCTSNode
        self.visits = 0
        self.q_values = defaultdict(float)
        self.action_visits = defaultdict(int)

    def is_fully_expanded(self):
        return len(self.children) == len(ALL_ACTIONS)

    def best_action(self, c=1.4):
        best_score = -float('inf')
        best_action = None
        for a in ALL_ACTIONS:
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

# MCTS主类
class MCTS:
    def __init__(self, transition_model, simulations=100, max_depth=5):
        self.P = transition_model
        self.simulations = simulations
        self.max_depth = max_depth

    def search(self, current_state):
        self.root = MCTSNode(current_state)

        for _ in range(self.simulations):
            self.simulate(self.root, depth=0)

        best_a = max(self.root.q_values.items(), key=lambda x: x[1])[0]
        return best_a

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

        # 回传更新
        node.visits += 1
        node.action_visits[action] += 1
        node.q_values[action] += (reward - node.q_values[action]) / node.action_visits[action]
        return reward

    def reward(self, state):
        return 1.0 if state == STATE_NORMAL else -1.0

# 示例主循环（用于实时决策）
if __name__ == "__main__":
    P_lib = TransitionProbability()
    mcts = MCTS(P_lib, simulations=50, max_depth=4)

    # 假设初始状态为注意力异常
    current_state = STATE_ATTENTION_ABNORMAL

    for step in range(10):
        print(f"\nStep {step}, 当前状态: {current_state}")
        best_action = mcts.search(current_state)
        print(f"推荐动作: {best_action}")

        # 模拟动作执行后的状态变化
        next_state = P_lib.sample(current_state, best_action)
        print(f"执行动作后状态: {next_state}")

        # 更新概率库
        P_lib.update(current_state, best_action, next_state)
        current_state = next_state
