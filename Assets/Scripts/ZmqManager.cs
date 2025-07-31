using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System.Text;

public class ZmqManager : MonoBehaviour
{
    // CursorData 类用于解析接收到的光标数据
    public class CursorData
    {
        public float x;
        public float y;
    }
    public GameObject cursorGameObject;

    // 网络套接字
    private SubscriberSocket _subSocket;
    private PublisherSocket _publisherSocket;
    
    // 初始化状态
    private bool _isReceiverInitialized = false;
    private bool _isPublisherInitialized = false;
    private bool _isPublisherRunning = true;
    
    // 副目标坐标
    private float secX = 4.0f, secY = 0.0f;
    
    // 引用TargetControl脚本来获取secX和secY值
    private TargetControl _targetControl;

    void Start()
    {
        InitializeReceiver();
        InitializePublisher();
    }

    // 初始化接收器（接收光标位置）
    private void InitializeReceiver()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();
            _subSocket = new SubscriberSocket();
            _subSocket.Connect("tcp://127.0.0.1:6001"); // 与 Python PUB 端口一致
            _subSocket.SubscribeToAnyTopic();
            _subSocket.Options.ReceiveHighWatermark = 1;  // 限制接收队列
            _isReceiverInitialized = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Cursor Receiver Socket 初始化失败: {e.Message}");
        }
    }

    // 初始化发布器（发布副目标位置）
    private void InitializePublisher()
    {
        // 获取TargetControl组件
        _targetControl = GetComponent<TargetControl>();
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
        // 处理光标位置接收
        if (_isReceiverInitialized)
        {
            ProcessMessages();
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
    }

    // 处理接收到的消息（光标位置）
    void ProcessMessages()
    {
        string latestMessage = null;
        
        // 接收所有可用消息，只保留最新的
        while (_subSocket.TryReceiveFrameString(out string json))
        {
            latestMessage = json;
        }
        
        // 只处理最新的消息
        if (!string.IsNullOrEmpty(latestMessage))
        {
            try
            {
                var data = JsonConvert.DeserializeObject<CursorData>(latestMessage);
                if (data != null)
                {
                    Vector3 pos = new Vector3(data.x, data.y, 0);
                    UpdateCursorVisual(pos);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"JSON 解析错误: {e.Message}");
            }
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
            // 清理订阅套接字
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