using System.Collections;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Collections.Concurrent;
using System.Threading;
using System.Text;

public class TargetControl : MonoBehaviour
{
    public float radius = 4f; // 初始半径
    public float initialSpeed = 40f; // 初始角速度
    public float speedChangeRate = 30f; // 速度变化率
    public float speedChangeFrequency = 3f; // 速度变化频率
    public GameObject secondaryObject; // 第二个绕圈运动的物体
    public float secondaryRadius = 4f; // 固定半径
    public float appearInterval = 2f;    // 出现时间间隔
    public float disappearDuration = 1f; // 消失持续时间
    public float RChangeInterval = 5f; // 触发频率（秒）


    private float angle = 0f; // 当前角度
    private float currentSpeed;
    private float targetRadius; // 目标半径
    private bool isRadiusChanging = false; // 是否正在改变半径
    private Coroutine radiusChangeCoroutine;

    private struct TargetData
    {
        public float x;
        public float y;
    }

    [Header("Send Settings")]
    [Tooltip("发送间隔（秒）")]
    [SerializeField] private float _sendInterval = 0.02f; // 50Hz

    private PushSocket _pushSocket;
    private Thread _sendThread;
    private bool _isSending = true;
    private ConcurrentQueue<TargetData> _sendQueue = new ConcurrentQueue<TargetData>();
    private float _lastSendTime = 0f;
    private float secX = 4.0f, secY = 0.0f;

    void Start()
    {
        // 初始化网络发送
        AsyncIO.ForceDotNet.Force();
        _pushSocket = new PushSocket();
        _pushSocket.Connect("tcp://127.0.0.1:6005"); // 使用新端口，避免与主目标冲突

        _sendThread = new Thread(SendSecondaryPositions);
        _sendThread.Start();

        currentSpeed = initialSpeed;
        targetRadius = radius; // 初始目标半径
        StartCoroutine(SecondaryObjectBlink());
        radiusChangeCoroutine = StartCoroutine(AutoRadiusChange());

    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            TriggerRadiusChange(3f); // 在 3 秒内进行半径变化
        }
        // 计算新的角度
        angle -= currentSpeed * Time.deltaTime;
        angle = (angle % 360f + 360f) % 360f;

        // 计算主物体的位置
        float x = Mathf.Cos(Mathf.Deg2Rad * angle) * radius;
        float y = Mathf.Sin(Mathf.Deg2Rad * angle) * radius;
        transform.position = new Vector3(x, y, transform.position.z);

        // 计算副物体的位置（固定半径）
        if (secondaryObject != null)
        {
            secX = Mathf.Cos(Mathf.Deg2Rad * angle) * secondaryRadius;
            secY = Mathf.Sin(Mathf.Deg2Rad * angle) * secondaryRadius;
            secondaryObject.transform.position = new Vector3(secX, secY, secondaryObject.transform.position.z);
        }

        // 改变角速度
        currentSpeed = initialSpeed + Mathf.Cos(Mathf.Deg2Rad * angle * speedChangeFrequency) * speedChangeRate;

        if (secondaryObject != null && secondaryObject.activeSelf)
        {
            secX = Mathf.Cos(Mathf.Deg2Rad * angle) * secondaryRadius;
            secY = Mathf.Sin(Mathf.Deg2Rad * angle) * secondaryRadius;
            secondaryObject.transform.position = new Vector3(secX, secY, secondaryObject.transform.position.z);
        }

        // 新增：按指定频率发送副目标位置
        if (Time.time - _lastSendTime >= _sendInterval && secondaryObject != null)
        {
            Vector2 secondaryPos = new Vector2(secX, secY);
            SendSecondaryPosition(secondaryPos);
            _lastSendTime = Time.time;
        }
    }

    public void TriggerRadiusChange(float T)
    {
        if (!isRadiusChanging)
        {
            // 随机选择区间后生成半径
            float newRadius = Random.Range(0, 2) == 0 ? Random.Range(2f, 3f) : Random.Range(5f, 6f);
            StartCoroutine(ChangeRadiusSmoothly(newRadius, T));
        }
    }

    private IEnumerator ChangeRadiusSmoothly(float newRadius, float duration)
    {
        isRadiusChanging = true;
        float elapsedTime = 0f;
        float startRadius = radius;

        // 平滑过渡到新半径
        while (elapsedTime < duration / 2f)
        {
            radius = Mathf.Lerp(startRadius, newRadius, elapsedTime / (duration / 2f));
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        radius = newRadius;
        elapsedTime = 0f;

        // while (elapsedTime < duration / 5f)
        // {
        //     // radius = Mathf.Lerp(newRadius, startRadius, elapsedTime / (duration / 3f));
        //     elapsedTime += Time.deltaTime;
        //     yield return null;
        // }
        // elapsedTime = 0f;

        // 平滑过渡回原半径
        while (elapsedTime < duration / 2f)
        {
            radius = Mathf.Lerp(newRadius, startRadius, elapsedTime / (duration / 2f));
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        radius = startRadius;

        isRadiusChanging = false;
    }

    private IEnumerator SecondaryObjectBlink()
    {
        while (true)
        {
            if (secondaryObject != null)
            {
                // 显示物体
                secondaryObject.SetActive(true);

                // 保持显示状态
                yield return new WaitForSeconds(appearInterval);

                // 隐藏物体
                secondaryObject.SetActive(false);

                // 保持隐藏状态
                yield return new WaitForSeconds(disappearDuration);
            }
            else
            {
                yield return null;
            }
        }
    }
    private IEnumerator AutoRadiusChange()
    {
        while (true)
        {
            yield return new WaitForSeconds(RChangeInterval);
            TriggerRadiusChange(3f); // 保持3秒的过渡时间
        }
    }

    // 新增：发送副目标位置的线程方法
    void SendSecondaryPositions()
    {
        var jsonBuilder = new StringBuilder(64);

        while (_isSending)
        {
            if (_sendQueue.TryDequeue(out TargetData data))
            {
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

    // 新增：将副目标位置加入发送队列
    public void SendSecondaryPosition(Vector2 position)
    {
        var data = new TargetData
        {
            x = position.x,
            y = position.y
        };

        _sendQueue.Enqueue(data);
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
        _isSending = false;
        if (_sendThread != null && _sendThread.IsAlive)
        {
            _sendThread.Join(100); // 等待100ms线程退出
        }
        _pushSocket?.Close();
        NetMQConfig.Cleanup();
        Debug.Log("SecondaryTargetSender 资源已清理");
    }
}
