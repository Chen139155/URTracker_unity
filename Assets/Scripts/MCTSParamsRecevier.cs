using UnityEngine;
using NetMQ; // 需要导入NetMQ插件
using NetMQ.Sockets;
using System.Threading;
using System.Collections.Concurrent;

public class MCTSParamsReceiver : MonoBehaviour
{
    private Thread _receiverThread;
    private bool _running = true;
    private readonly ConcurrentQueue<MCTSParams> _paramsQueue = new ConcurrentQueue<MCTSParams>();
    
    // 参数结构体（与Python端对应）
    public struct MCTSParams
    {
        public int diffculty;  // 注意保持拼写一致（Python端为diffculty）
        public int feedback;
        public int assistance;
    }

    void Start()
    {
        _receiverThread = new Thread(ReceiveMessages);
        _receiverThread.Start();
    }

    private void ReceiveMessages()
    {
        using (var subSocket = new SubscriberSocket())
        {
            subSocket.Options.ReceiveHighWatermark = 1000;
            subSocket.Connect("tcp://localhost:5557"); // 与Python端pub端口一致
            subSocket.Subscribe("mcts_params");

            while (_running)
            {
                if (subSocket.TryReceiveFrameString(out string topic) && 
                    subSocket.TryReceiveFrameString(out string json))
                {
                    if (topic == "mcts_params")
                    {
                        var data = JsonUtility.FromJson<MCTSParams>(json);
                        _paramsQueue.Enqueue(data);
                    }
                }
                else
                {
                    Thread.Sleep(10); // 避免CPU空转
                }
            }
        }
    }

    void Update()
    {
        // 在主线程处理接收到的参数
        if (_paramsQueue.TryDequeue(out MCTSParams params))
        {
            UpdateUnityVisualization(params);
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
        _running = false;
        _receiverThread?.Join();
    }
}