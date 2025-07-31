
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Text;


public class SecondaryTargetPublisher : MonoBehaviour
{
    private PublisherSocket _publisherSocket;
    private bool _isInitialized = false;
    private bool _isRunning = true;
    private float secX = 4.0f, secY = 0.0f;
    
    // 引用TargetControl脚本来获取secX和secY值
    private TargetControl _targetControl;

    void Start()
    {
        // 获取TargetControl组件
        _targetControl = GetComponent<TargetControl>();
        if (_targetControl == null)
        {
            Debug.LogError("未找到 TargetControl 组件！");
            return;
        }

        // 初始化网络发送
        try
        {
            AsyncIO.ForceDotNet.Force();
            _publisherSocket = new PublisherSocket();
            _publisherSocket.Connect("tcp://localhost:6005"); // 使用6005端口发布
            _isInitialized = true;
            _isRunning = true;
            Debug.Log("SecondaryTargetPublisher 初始化成功，连接到 tcp://localhost:6005");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Publisher Socket 初始化失败: {e.Message}");
        }
    }

    void Update()
    {
        if (!_isInitialized || !_isRunning) return;

        // 获取TargetControl中的secX和secY值
        if (_targetControl != null)
        {
            secX = _targetControl.SecX;
            secY = _targetControl.SecY;
        }

        Vector2 secondaryPos = new Vector2(secX, secY);
        PublishSecondaryPosition(secondaryPos);
    }

    // 发布副目标位置
    public void PublishSecondaryPosition(Vector2 position)
    {
        if (!_isInitialized || !_isRunning) return;

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
        _isRunning = false;
        CleanupResources();
    }
    
    private void CleanupResources()
    {
        _isInitialized = false;
        
        try
        {
            if (_publisherSocket != null)
            {
                // 先断开连接再关闭
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
        
        Debug.Log("SecondaryTargetPublisher 资源已清理");
    }
}