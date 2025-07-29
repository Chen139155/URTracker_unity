// UnityCursorReceiver.cs
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;

public class UnityCursorReceiver : MonoBehaviour
{
    public class CursorData
    {
        public float x; // 必须与 Python 字段名完全一致
        public float y;
    }

    private SubscriberSocket _subSocket;
    private bool _isInitialized = false;

    void Start()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();
            _subSocket = new SubscriberSocket();
            _subSocket.Connect("tcp://127.0.0.1:6001"); // 与 Python PUB 端口一致
            _subSocket.SubscribeToAnyTopic();
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

    void ProcessMessages()
    {
        // 处理所有可用的消息，但限制处理数量以避免帧率下降
        int messageCount = 0;
        const int maxMessagesPerFrame = 10;
        
        while (messageCount < maxMessagesPerFrame && _subSocket.TryReceiveFrameString(out string json))
        {
            try
            {
                var data = JsonConvert.DeserializeObject<CursorData>(json);
                if (data != null)
                {
                    Vector3 pos = new Vector3(
                        data.x,
                        data.y,
                        0
                    );
                    UpdateCursorVisual(pos);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"JSON 解析错误: {e.Message}");
            }
            
            messageCount++;
        }
    }

    void UpdateCursorVisual(Vector3 position)
    {
        transform.position = position;
        // 移除了调试日志以减少输出，如需要可取消注释下一行
        Debug.Log($"更新光标位置: {position}");
    }

    void OnDestroy()
    {
        _isInitialized = false;
        _subSocket?.Close();
        NetMQConfig.Cleanup();
    }
    
    void OnApplicationQuit()
    {
        _isInitialized = false;
        _subSocket?.Close();
        NetMQConfig.Cleanup();
        Debug.Log("UnityCursorReceiver 资源已清理");
    }
}