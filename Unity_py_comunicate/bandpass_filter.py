# 在文件开头添加导入
from scipy import signal
import numpy as np
import logging

def robust_bandpass_filter(data, lowcut=0.1, highcut=10.0, fs=50.0, method='standard'):
    """
    健壮的带通滤波器，处理边界效应
    
    参数:
    data: 输入数据
    lowcut, highcut: 截止频率
    fs: 采样频率
    method: 滤波方法 ('standard', 'causal', 'mirror', 'adaptive')
    """
    if len(data) < 5:
        return data
    
    try:
        if method == 'standard':
            # 标准零相位滤波
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.filtfilt(b, a, data)
            
        elif method == 'causal':
            # 因果滤波（单向）
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.lfilter(b, a, data)
            
        elif method == 'mirror':
            # 镜像扩展
            return mirror_extension_filter(data, lowcut, highcut, fs)
            
        elif method == 'adaptive':
            # 自适应滤波
            return adaptive_bandpass_filter(data, lowcut, highcut, fs)
            
    except Exception as e:
        logging.warning(f"滤波器应用失败: {e}")
        # 失败时返回原始数据或简单平滑数据
        if len(data) > 3:
            window_size = min(3, len(data))
            if window_size % 2 == 0:
                window_size -= 1
            if window_size > 1:
                return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return data

# 在主代码中使用
# recent_data = df_window[-50:] if len(df_window) >= 50 else df_window
# target_x_data = np.array([row["target_x"] for row in recent_data])
# filtered_target_x = robust_bandpass_filter(target_x_data, 0.1, 10.0, 50.0, method='mirror')

def adaptive_bandpass_filter(data, lowcut, highcut, fs):
    """
    自适应带通滤波器，根据数据长度调整参数
    
    参数:
    data: 输入数据
    lowcut, highcut: 截止频率
    fs: 采样频率
    """
    from scipy import signal
    
    if len(data) < 20:
        # 数据太短，使用简单平滑
        window_size = min(5, len(data))
        if window_size % 2 == 0:
            window_size -= 1  # 确保奇数窗口
        if window_size > 1:
            filtered_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
            return filtered_data
        else:
            return data
    
    # 根据数据长度调整滤波器参数
    order = min(4, max(2, len(data) // 10))  # 动态调整阶数
    
    # 设计滤波器
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # 使用椭圆滤波器（更好的频率响应）
    b, a = signal.ellip(order, 0.5, 40, [low, high], btype='band')
    
    # 应用滤波器
    try:
        filtered_data = signal.filtfilt(b, a, data)
    except:
        # 如果滤波失败，使用简单移动平均
        window_size = min(5, len(data) // 2)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            filtered_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        else:
            filtered_data = data
    
    return filtered_data

def mirror_extension_filter(data, lowcut, highcut, fs, extension_factor=2):
    """
    通过镜像扩展数据来减少边界效应
    
    参数:
    data: 输入数据
    lowcut, highcut: 截止频率
    fs: 采样频率
    extension_factor: 扩展因子
    """
    from scipy import signal
    
    if len(data) < 10:
        return data
    
    # 镜像扩展数据
    extension_len = min(len(data) // extension_factor, 20)
    extended_data = np.concatenate([
        np.flip(data[1:extension_len+1]),  # 前部镜像
        data,                              # 原始数据
        np.flip(data[-extension_len-1:-1]) # 后部镜像
    ])
    
    # 设计滤波器
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # 滤波
    filtered_extended = signal.filtfilt(b, a, extended_data)
    
    # 提取原始长度的数据（去除扩展部分）
    result = filtered_extended[extension_len:-extension_len]
    
    return result