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
     
    private PushSocket _pushSocket;
    private Thread _sendThread;
    private bool _isSending = true;
    private ConcurrentQueue<TargetData> _sendQueue = new ConcurrentQueue<TargetData>();
    // private int _syncCounter = 0;
    private float _lastSendTime = 0f;

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _pushSocket = new PushSocket();
        _pushSocket.Connect("tcp://127.0.0.1:6000"); // 与 Python PULL 端口一致

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

                _pushSocket.SendFrame(jsonBuilder.ToString());
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
        _pushSocket?.Close();
        NetMQConfig.Cleanup();
    }
}