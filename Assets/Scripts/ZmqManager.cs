using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System.Text;
using System.Collections.Generic;
using TMPro;
using UnityEngine.UI;

public class ZmqManager : MonoBehaviour
{
    // CursorData 类用于解析接收到的光标数据
    public class CursorData
    {
        public float x;
        public float y;
    }
    
    // MetricsData 类用于解析接收到的指标数据
    public class MetricsData
    {
        public float e_c;
        public float e_m;
        public double timestamp;
    }
    
    // MCTSActionData 类用于解析接收到的MCTS动作数据
    public class MCTSActionData
    {
        public string difficulty;
        public string feedback;
        public string assistance;
        public double timestamp;
    }
    
    public GameObject cursorGameObject;
    public Slider metricsBar_e_c;
    public Slider metricsBar_e_m;
    public TextMeshProUGUI mctsActionText;

    // 网络套接字
    private SubscriberSocket _subSocket;          // 用于接收光标位置
    private PublisherSocket _publisherSocket;
    
    // 初始化状态
    private bool _isReceiverInitialized = false;
    private bool _isPublisherInitialized = false;
    private bool _isPublisherRunning = true;
    
    // 副目标坐标
    private float secX = 4.0f, secY = 0.0f;
    
    // 引用TargetControl脚本来获取secX和secY值
    private TargetControl _targetControl;
    
    // 用于存储接收到的数据
    private MetricsData _latestMetrics;
    private MCTSActionData _latestMCTSAction;
    
    // 用于平滑过渡的变量
    private float _targetEC = 0f;
    private float _targetEM = 0f;
    private float _currentEC = 0f;
    private float _currentEM = 0f;
    private bool _isSmoothing = false;
    private float _smoothingSpeed = 5.0f; // 平滑过渡速度

    void Start()
    {
        _targetControl = GetComponent<TargetControl>();
        InitializeReceiver();
        InitializePublisher();
        // 获取TargetControl组件
        
        // 初始化当前值
        if (metricsBar_e_c != null)
        {
            _currentEC = metricsBar_e_c.value;
            _targetEC = _currentEC;
        }
        if (metricsBar_e_m != null)
        {
            _currentEM = metricsBar_e_m.value;
            _targetEM = _currentEM;
        }
    }

    // 初始化接收器（接收光标位置）
    private void InitializeReceiver()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();
            _subSocket = new SubscriberSocket();
            _subSocket.Connect("tcp://127.0.0.1:6001"); // 与 Python PUB 端口一致
            _subSocket.Subscribe("cursor");     // 订阅光标数据
            _subSocket.Subscribe("metrics");    // 订阅指标数据
            _subSocket.Subscribe("mcts_action"); // 订阅MCTS动作数据
            _subSocket.Options.ReceiveHighWatermark = 10;  // 增加接收队列
            _isReceiverInitialized = true;
            Debug.Log("Unified Receiver 初始化成功");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Receiver Socket 初始化失败: {e.Message}");
        }
    }

    // 初始化发布器（发布副目标位置）
    private void InitializePublisher()
    {
        
        if (_targetControl == null)
        {
            Debug.LogError("未找到 TargetControl 组件！");
            return;
        }

        try
        {
            AsyncIO.ForceDotNet.Force();
            _publisherSocket = new PublisherSocket();
            _publisherSocket.Connect("tcp://localhost:6005"); // 使用6005端口发布
            _isPublisherInitialized = true;
            _isPublisherRunning = true;
            Debug.Log("SecondaryTargetPublisher 初始化成功，连接到 tcp://localhost:6005");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Publisher Socket 初始化失败: {e.Message}");
        }
    }

    void Update()
    {
        // 处理所有消息接收
        if (_isReceiverInitialized)
        {
            ProcessAllMessages();
        }

        // 处理副目标位置发布
        if (_isPublisherInitialized && _isPublisherRunning)
        {
            // 获取TargetControl中的secX和secY值
            if (_targetControl != null)
            {
                secX = _targetControl.SecX;
                secY = _targetControl.SecY;
            }

            Vector2 secondaryPos = new Vector2(secX, secY);
            PublishSecondaryPosition(secondaryPos);
        }
        
        // 平滑更新 Metrics 量条
        SmoothUpdateMetricsBars();

        // 更新 MCTS 动作显示
        if (_latestMCTSAction != null)
        {
            mctsActionText.text = $"Difficulty: {_latestMCTSAction.difficulty}\nFeedback: {_latestMCTSAction.feedback}\nAssistance: {_latestMCTSAction.assistance}";
        }
    }

    // 统一处理所有消息
    void ProcessAllMessages()
    {
        List<string> message=null;
        
        // 接收所有可用消息
        while (_subSocket.TryReceiveMultipartStrings(ref message))
        {
            if (message.Count >= 2)
            {
                string topic = message[0];
                string jsonData = message[1];
                
                try
                {
                    switch (topic)
                    {
                        case "cursor":
                            var cursorData = JsonConvert.DeserializeObject<CursorData>(jsonData);
                            if (cursorData != null)
                            {
                                Vector3 pos = new Vector3(cursorData.x, cursorData.y, 0);
                                UpdateCursorVisual(pos);
                            }
                            break;
                            
                        case "metrics":
                            var metricsData = JsonConvert.DeserializeObject<MetricsData>(jsonData);
                            if (metricsData != null)
                            {
                                _latestMetrics = metricsData;
                                // 设置目标值用于平滑过渡
                                if (metricsBar_e_c != null)
                                {
                                    _targetEC = Mathf.Clamp(metricsData.e_c, metricsBar_e_c.minValue, metricsBar_e_c.maxValue);
                                }
                                if (metricsBar_e_m != null)
                                {
                                    _targetEM = Mathf.Clamp(metricsData.e_m, metricsBar_e_m.minValue, metricsBar_e_m.maxValue);
                                }
                                _isSmoothing = true;
                                Debug.Log($"接收到指标数据: e_c={metricsData.e_c:F4}, e_m={metricsData.e_m:F4}");
                            }
                            break;
                            
                        case "mcts_action":
                            var actionData = JsonConvert.DeserializeObject<MCTSActionData>(jsonData);
                            if (actionData != null)
                            {
                                _latestMCTSAction = actionData;
                                Debug.Log($"接收到MCTS动作: 难度={actionData.difficulty}, 反馈={actionData.feedback}, 辅助={actionData.assistance}");
                                _targetControl.SetDifficulty(actionData.difficulty);
                                _targetControl.SetFeedbackLevel(actionData.feedback);
                            }
                            break;
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"JSON 解析错误 (Topic: {topic}): {e.Message}");
                }
            }
        }
    }

    // 平滑更新指标量条
    private void SmoothUpdateMetricsBars()
    {
        if (!_isSmoothing) return;
        
        bool ecUpdated = false;
        bool emUpdated = false;
        
        // 平滑更新 e_c 量条
        if (metricsBar_e_c != null && Mathf.Abs(_currentEC - _targetEC) > 0.001f)
        {
            _currentEC = Mathf.Lerp(_currentEC, _targetEC, Time.deltaTime * _smoothingSpeed);
            metricsBar_e_c.value = _currentEC;
            ecUpdated = true;
        }
        else if (metricsBar_e_c != null)
        {
            _currentEC = _targetEC;
            metricsBar_e_c.value = _currentEC;
        }
        
        // 平滑更新 e_m 量条
        if (metricsBar_e_m != null && Mathf.Abs(_currentEM - _targetEM) > 0.001f)
        {
            _currentEM = Mathf.Lerp(_currentEM, _targetEM, Time.deltaTime * _smoothingSpeed);
            metricsBar_e_m.value = _currentEM;
            emUpdated = true;
        }
        else if (metricsBar_e_m != null)
        {
            _currentEM = _targetEM;
            metricsBar_e_m.value = _currentEM;
        }
        
        // 如果两个量条都已达到目标值，停止平滑更新
        if (!ecUpdated && !emUpdated)
        {
            _isSmoothing = false;
        }
    }

    // 更新光标视觉表现
    void UpdateCursorVisual(Vector3 position)
    {
        if (cursorGameObject != null)
        {
            cursorGameObject.transform.position = position;
        }
        // transform.position = position;
        // 移除了调试日志以减少输出，如需要可取消注释下一行
        // Debug.Log($"更新光标位置: {position}");
    }

    // 发布副目标位置
    public void PublishSecondaryPosition(Vector2 position)
    {
        if (!_isPublisherInitialized || !_isPublisherRunning) return;

        try
        {
            var jsonBuilder = new StringBuilder(64);
            jsonBuilder.Clear();
            jsonBuilder.Append("{\"x\":")
                       .Append(position.x.ToString("F4")).Append(",")
                       .Append("\"y\":")
                       .Append(position.y.ToString("F4"))
                       .Append("}");

            // 使用非阻塞发送避免潜在的阻塞问题
            _publisherSocket.TrySendFrame(jsonBuilder.ToString());
        }
        catch (System.Exception e)
        {
            Debug.LogError($"发布目标位置失败: {e.Message}");
        }
    }
    
    void OnDisable()
    {
        StopAllOperations();
    }
    
    void OnDestroy()
    {
        StopAllOperations();
    }
    
    void OnApplicationQuit()
    {
        StopAllOperations();
    }
    
    private void StopAllOperations()
    {
        _isPublisherRunning = false;
        CleanupResources();
    }
    
    private void CleanupResources()
    {
        _isReceiverInitialized = false;
        _isPublisherInitialized = false;
        
        try
        {
            // 清理光标订阅套接字
            if (_subSocket != null)
            {
                _subSocket.Close();
                _subSocket = null;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"关闭SubscriberSocket时出错: {e.Message}");
        }
        try
        {
            // 清理发布套接字
            if (_publisherSocket != null)
            {
                _publisherSocket.Dispose();
                _publisherSocket = null;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"关闭PublisherSocket时出错: {e.Message}");
        }
        
        try
        {
            // 清理NetMQ配置
            NetMQConfig.Cleanup();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"NetMQ清理时出错: {e.Message}");
        }
        
        Debug.Log("UnityCursorAndTargetHandler 资源已清理");
    }
}