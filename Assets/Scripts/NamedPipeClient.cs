using UnityEngine;
using System.IO;
using System.IO.Pipes;
using System.Threading;
using System.Collections.Concurrent;

public class NamedPipeClient : MonoBehaviour
{
    public GameObject targetGameObject; // 目标物体（在 Inspector 里赋值）
    public GameObject cursorGameObject;
    private Thread targetThread, cursorThread;
    private bool running = true;
    private string targetPipeName = "UnityToPythonPipe";
    private string cursorPipeName = "PythonToUnityPipe";
    private ConcurrentQueue<Vector2> cursorQueue = new ConcurrentQueue<Vector2>();
    private Vector2 latestCursor = Vector2.zero;
    private Vector3 targetPosition = Vector3.zero;

    void Start()
    {
        targetThread = new Thread(TargetPipeWorker);
        cursorThread = new Thread(CursorPipeWorker);
        targetThread.Start();
        cursorThread.Start();
    }

    // 目标坐标发送线程
    void TargetPipeWorker()
    {
        while (running)
        {
            using (NamedPipeClientStream pipe = ConnectPipe(targetPipeName, PipeDirection.Out))
            {
                if (pipe == null) continue;
                using (StreamWriter writer = new StreamWriter(pipe) { AutoFlush = true })
                {
                    while (running)
                    {
                        SendTargetCoordinates(writer);
                        Thread.Sleep(16); // 60Hz 发送
                    }
                }
            }
            Thread.Sleep(1000); // 断开后 1 秒重试
        }
    }

    // 光标坐标接收线程
    void CursorPipeWorker()
    {
        while (running)
        {
            using (NamedPipeClientStream pipe = ConnectPipe(cursorPipeName, PipeDirection.In))
            {
                if (pipe == null) continue;
                using (StreamReader reader = new StreamReader(pipe))
                {
                    while (running)
                    {
                        ReadCursorCoordinates(reader);
                    }
                }
            }
            Thread.Sleep(1000); // 断开后 1 秒重试
        }
    }

    // 连接管道（失败时返回 null）
    NamedPipeClientStream ConnectPipe(string pipeName, PipeDirection direction)
    {
        try
        {
            NamedPipeClientStream pipe = new NamedPipeClientStream(".", pipeName, direction);
            pipe.Connect(5000); // 5 秒超时
            Debug.Log($"[Unity] 连接管道 {pipeName} 成功");
            return pipe;
        }
        catch (IOException)
        {
            Debug.LogWarning($"[Unity] 连接管道 {pipeName} 失败，1 秒后重试...");
            return null;
        }
    }

    // 发送 Target 坐标
    void SendTargetCoordinates(StreamWriter writer)
    {
        writer.WriteLine($"{targetPosition.x},{targetPosition.y}");
        Debug.Log($"[Unity] 发送目标坐标: {targetPosition.x}, {targetPosition.y}");
    }

    // 读取 Cursor 坐标
    void ReadCursorCoordinates(StreamReader reader)
    {
        string line = reader.ReadLine();
        if (!string.IsNullOrEmpty(line))
        {
            string[] parts = line.Split(',');
            if (parts.Length == 2 && float.TryParse(parts[0], out float x) && float.TryParse(parts[1], out float y))
            {
                Vector2 newCursor = new Vector2(x, y);
                cursorQueue.Enqueue(newCursor);
                Debug.Log($"[Unity] 接收光标坐标: {x}, {y }");
            }
        }
    }

    void Update()
    {
        if (targetGameObject != null)
        {
            targetPosition = targetGameObject.transform.position;
        }
        if (cursorQueue.TryDequeue(out Vector2 coord))
        {
            latestCursor = coord;
        }
        
        cursorGameObject.transform.position = new Vector3(latestCursor.x, latestCursor.y, 0); // 移动游戏对象
    }

    void OnApplicationQuit()
    {
        running = false;
        targetThread.Join();
        cursorThread.Join();
    }
}
