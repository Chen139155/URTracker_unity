using UnityEngine;
using NetMQ; // 需要导入NetMQ插件
using NetMQ.Sockets;

public class MCTSParamsReceiver : MonoBehaviour
{
    private SubscriberSocket _subSocket;
    private bool _isInitialized = false;
    
    // 参数结构体（与Python端对应）
    public struct MCTSParams
    {
        public int diffculty;  // 注意保持拼写一致（Python端为diffculty）
        public int feedback;
        public int assistance;
    }

    void Start()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();
            _subSocket = new SubscriberSocket();
            _subSocket.Options.ReceiveHighWatermark = 1000;
            _subSocket.Connect("tcp://localhost:5557"); // 与Python端pub端口一致
            _subSocket.Subscribe("mcts_params");
            _isInitialized = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Socket 初始化失败: {e.Message}");
        }
    }

    void Update()
    {
        if (!_isInitialized) return;

        // 在主线程中处理消息接收
        ProcessMessages();
    }

    private void ProcessMessages()
    {
        // 处理所有可用的消息，但限制处理数量以避免帧率下降
        int messageCount = 0;
        const int maxMessagesPerFrame = 10;
        
        while (messageCount < maxMessagesPerFrame && 
               _subSocket.TryReceiveFrameString(out string topic) && 
               _subSocket.TryReceiveFrameString(out string json))
        {
            if (topic == "mcts_params")
            {
                try
                {
                    var data = JsonUtility.FromJson<MCTSParams>(json);
                    UpdateUnityVisualization(data);
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"JSON 解析错误: {e.Message}");
                }
            }
            
            messageCount++;
        }
    }

    private void UpdateUnityVisualization(MCTSParams parameters)
    {
        // 这里实现参数到Unity可视化元素的映射逻辑，例如：
        // 1. 根据difficulty调整场景复杂度
        // 2. 根据feedback调整反馈强度
        // 3. 根据assistance调整辅助等级
        Debug.Log($"Received: {parameters.diffculty}, {parameters.feedback}, {parameters.assistance}");
    }

    void OnDestroy()
    {
        CleanupResources();
    }
    
    void OnApplicationQuit()
    {
        CleanupResources();
    }
    
    private void CleanupResources()
    {
        _isInitialized = false;
        
        if (_subSocket != null)
        {
            _subSocket.Close();
        }
        
        NetMQConfig.Cleanup();
        Debug.Log("MCTSParamsReceiver 资源已清理");
    }
}