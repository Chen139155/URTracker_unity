// UnityTargetSender.cs
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Collections.Concurrent;
using System.Threading;
using System.Text;

public class UnityTargetSender : MonoBehaviour
{
    private struct TargetData
    {
        public float x; // 必须与 Python 字段名完全一致
        public float y;
    }

    [Header("Send Settings")]
    [Tooltip("发送间隔（秒）")]
    [SerializeField] private float _sendInterval = 0.02f; // 50Hz

    private PublisherSocket _publisherSocket;  // 修改1: PushSocket -> PublisherSocket
    private Thread _sendThread;
    private bool _isSending = true;
    private ConcurrentQueue<TargetData> _sendQueue = new ConcurrentQueue<TargetData>();
    private float _lastSendTime = 0f;
 
    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _publisherSocket = new PublisherSocket();  // 修改2: 实例化 PublisherSocket
        _publisherSocket.Connect("tcp://127.0.0.1:6005"); // 与 Python SUB 端口一致

        _sendThread = new Thread(SendTargets);
        _sendThread.Start();
    }

    void Update()
    {
        // 按指定频率发送当前位置
        if (Time.time - _lastSendTime >= _sendInterval)
        {
            SendTargetPosition(transform.position);
            _lastSendTime = Time.time;
        }
    }

    void SendTargets()
    {
        var jsonBuilder = new StringBuilder(64);

        while (_isSending)
        {
            if (_sendQueue.TryDequeue(out TargetData data))
            {
                // 手动构建 JSON 提升性能
                jsonBuilder.Clear();
                jsonBuilder.Append("{\"x\":")
                           .Append(data.x.ToString("F4")).Append(",")
                           .Append("\"y\":")
                           .Append(data.y.ToString("F4"))
                           .Append("}");

                _publisherSocket.SendFrame(jsonBuilder.ToString());  // 修改3: 使用 PublisherSocket 发送
            }
            else
            {
                Thread.Sleep(1);
            }
        }
    }

    public void SendTargetPosition(Vector3 target)
    {
        var data = new TargetData
        {
            x = target.x,
            y = target.y
        };

        _sendQueue.Enqueue(data);
    }

    void OnDestroy()
    {
        _isSending = false;
        if (_sendThread != null && _sendThread.IsAlive)
        {
            _sendThread.Join(100); // 等待100ms线程退出
        }
        _publisherSocket?.Close();  // 修改4: 关闭 PublisherSocket
        NetMQConfig.Cleanup();
    }
    
    void OnApplicationQuit()
    {
        _isSending = false;
        if (_sendThread != null && _sendThread.IsAlive)
        {
            _sendThread.Join(100); // 等待100ms线程退出
        }
        _publisherSocket?.Close();  // 修改5: 关闭 PublisherSocket
        NetMQConfig.Cleanup();
        Debug.Log("UnityTargetSender 资源已清理");
    }
}