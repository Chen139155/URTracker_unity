// UnityCursorReceiver.cs
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Collections.Concurrent;
using System.Threading;
using Newtonsoft.Json;

public class UnityCursorReceiver : MonoBehaviour
{
    public class CursorData
    {
        public float x; // 必须与 Python 字段名完全一致
        public float y;
    }

    private Thread _receiveThread;
    private bool _isRunning = true;
    private ConcurrentQueue<Vector3> _positionQueue = new ConcurrentQueue<Vector3>();
    private SubscriberSocket _subSocket;

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _subSocket = new SubscriberSocket();
        _subSocket.Connect("tcp://127.0.0.1:6001"); // 与 Python PUSH 端口一致
        _subSocket.SubscribeToAnyTopic();

        _receiveThread = new Thread(ReceiveCursor);
        _receiveThread.Start();
    }

    void ReceiveCursor()
    {
        while (_isRunning)
        {
            if (_subSocket.TryReceiveFrameString(out string json))
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
                        _positionQueue.Enqueue(pos);
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"JSON 解析错误: {e.Message}");
                }
            }
            Thread.Sleep(0);
        }
    }

    void Update()
    {
        while (_positionQueue.TryDequeue(out Vector3 pos))
        {
            UpdateCursorVisual(pos);
        }
    }

    void UpdateCursorVisual(Vector3 position)
    {
        transform.position = position;
        Debug.Log($"更新光标位置: {position}");
    }

    void OnDestroy()
    {
        _isRunning = false;
        _subSocket?.Close();
        NetMQConfig.Cleanup();
    }
    void OnApplicationQuit()
    {
        _isRunning = false;
        if (_receiveThread != null && _receiveThread.IsAlive)
        {
            _receiveThread.Join(100); // 等待100ms线程退出
        }
        _subSocket?.Close();
        NetMQConfig.Cleanup();
        Debug.Log("UnityCursorReceiver 资源已清理");
    }
}