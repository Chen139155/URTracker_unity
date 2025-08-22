import random
import math
from collections import defaultdict, namedtuple
import logging
import time
import queue
import json
import os

# ==== 状态定义 ====
STATE_LABELS = {
    0: 'Normal',
    1: 'Motor Abnormal',
    2: 'Attention Abnormal',
    3: 'Mixed Abnormal'
}

# ==== 动作定义 ====
DifficultyLevels = ['low', 'medium', 'high']
FeedbackLevels = ['high', 'medium', 'low']  # 越少越挑战
AssistanceLevels = ['high', 'medium', 'low']  # 越少越挑战

Action = namedtuple("Action", ["difficulty", "feedback", "assistance"])

ACTIONS = []
for d in DifficultyLevels:
    for f in FeedbackLevels:
        for a in AssistanceLevels:
            ACTIONS.append(Action(d, f, a))

# ==== 奖励函数 ====
def compute_reward(state, action):
    # 人的状态分值（正常最好）
    state_score = {
        0: 4.0,
        1: 0.5,
        2: 0.5,
        3: 0.0
    }[state]
    
    # 动作挑战性（越难、越少辅助/反馈 → 得分高）
    difficulty_score = {'low': 0.0, 'medium': 0.5, 'high': 1.0}[action.difficulty]
    feedback_score   = {'high': 0.0, 'medium': 0.5, 'low': 1.0}[action.feedback]
    assistance_score = {'high': 0.0, 'medium': 0.5, 'low': 1.0}[action.assistance]

    challenge_score = (difficulty_score + feedback_score + assistance_score) / 3.0

    # 奖励是两者结合，权重可调整
    return 0.75 * state_score + 0.25 * challenge_score

# ==== 状态转移概率库 ====
import json
import os

class TransitionModel:
    def __init__(self):
        # P(next_state | current_state, action)
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

    def save(self, filepath):
        # 转成可序列化格式
        serializable_counts = {}
        for (state, action), next_states in self.counts.items():
            key_str = f"{state}|{action.difficulty}|{action.feedback}|{action.assistance}"
            serializable_counts[key_str] = dict(next_states)
        with open(filepath, 'w') as f:
            json.dump(serializable_counts, f, indent=4)
        print(f"TransitionModel 已保存到 {filepath}")

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


# ==== MCTS节点与主类 ====
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
        max_depth = 5  # 增加模拟深度
        
        for _ in range(max_depth):
            action = random.choice(ACTIONS)
            next_state = self.transition_model.sample_next_state(current_state, action)
            reward = compute_reward(next_state, action)
            total_reward += reward * (0.9 ** depth)  # 折扣因子
            current_state = next_state
            depth += 1
            
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

def mcts_thread_function(mcts_input_queue, mcts_output_queue):
    """
    MCTS线程函数
    """
    # 初始化转移模型和MCTS
    transition_model = TransitionModel()
    # ==== 加载模型 ====
    model_file = "model.json"
    transition_model.load(model_file)

    mcts = MCTS(transition_model=transition_model, iterations=100)
    
    last_state = None
    last_action = None
    
    while True:
        try:
            # 获取最新的状态信息
            state_info = None
            try:
                while True:
                    state_info = mcts_input_queue.get_nowait()
            except queue.Empty:
                pass
            
            if state_info:
                state_flag = state_info['state_flag']
                e_c = state_info['e_c']
                e_m = state_info['e_m']
                timestamp = state_info['timestamp']
                
                # 更新转移模型（如果有前一个状态和动作）
                if last_state is not None and last_action is not None:
                    # # 根据e_c和e_m确定当前实际状态
                    # if e_c < 0.01 and e_m < 0.008:
                    #     actual_state = 0  # Normal
                    # elif e_c < 0.01 and e_m >= 0.008:
                    #     actual_state = 1  # Motor Abnormal
                    # elif e_c >= 0.01 and e_m < 0.008:
                    #     actual_state = 2  # Attention Abnormal
                    # else:
                    #     actual_state = 3  # Mixed Abnormal
                    
                    # 更新转移模型
                    transition_model.update(last_state, last_action, state_flag)
                
                # 执行MCTS搜索
                best_action_node = mcts.search(state_flag)
                best_action = best_action_node.action
                
                # 记录当前状态和动作，用于下次更新转移模型
                last_state = state_flag
                last_action = best_action
                
                # 发送最佳动作
                mcts_output_queue.put({
                    'action': best_action,
                    'timestamp': time.time()
                })
                
                logging.info(f"MCTS推荐动作: 难度={best_action.difficulty}, 反馈={best_action.feedback}, 辅助={best_action.assistance}")
            
            time.sleep(1.0)  # 每秒执行一次
            transition_model.save(model_file)
            
        except Exception as e:
            logging.error(f"MCTS线程错误: {e}")
            time.sleep(1.0)

if __name__ == '__main__':
    # 初始化转移模型和MCTS
    tm = TransitionModel()
    mcts = MCTS(transition_model=tm, iterations=200)

    # 假设从状态 S1（注意力异常）开始
    initial_state = 1
    best_next = mcts.search(initial_state)

    print(f"推荐动作: 难度={best_next.action.difficulty}, 反馈={best_next.action.feedback}, 辅助={best_next.action.assistance}")
    print(f"预期状态: {STATE_LABELS[best_next.state]}")
