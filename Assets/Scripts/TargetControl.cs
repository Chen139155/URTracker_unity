using System.Collections;
using UnityEngine;

public class TargetControl : MonoBehaviour
{
    // 难度设置类
    [System.Serializable]
    public class DifficultySettings
    {
        public float initialSpeed;
        public float speedChangeRate;
    }

    // 三档难度设置
    public DifficultySettings easyDifficulty = new DifficultySettings { initialSpeed = 20f, speedChangeRate = 15f };
    public DifficultySettings mediumDifficulty = new DifficultySettings { initialSpeed = 30f, speedChangeRate = 22f };
    public DifficultySettings hardDifficulty = new DifficultySettings { initialSpeed = 40f, speedChangeRate = 30f };

    public float radius = 4f; // 初始半径
    public float speedChangeFrequency = 3f; // 速度变化频率
    public float speedChangePhase = 0f; // 速度变化的相位
    public GameObject secondaryObject; // 第二个绕圈运动的物体
    public float secondaryRadius = 4f; // 固定半径
    public float appearInterval = 0.3f;    // 出现时间间隔
    public float disappearDuration = 0.7f; // 消失持续时间
    public float RChangeInterval = 5f; // 触发频率（秒）

    [Header("控制选项")]
    public bool enableRadiusChange = true; // 是否启用半径变化功能
    public StageFlowManager stageFlowManager; // 在Inspector中拖入StageFlowManager

    private float angle = 0f; // 当前角度
    private float currentSpeed;
    private float targetRadius; // 目标半径
    private bool isRadiusChanging = false; // 是否正在改变半径
    private Coroutine radiusChangeCoroutine;
    private Coroutine secondaryObjectBlinkCoroutine; // 保存协程引用
    private float timeElapsed = 0f; // 添加时间累计变量

    private float secX = 4.0f, secY = 0.0f;
    public float SecX { get { return secX; } }
    public float SecY { get { return secY; } }

    // 直接使用引用而不是复制值
    private DifficultySettings currentDifficulty_setting = new DifficultySettings { initialSpeed = 30f, speedChangeRate = 22f };

    // 当前难度级别
    private string currentDifficulty = "medium";
    
    // 是否开始运动的标志
    private bool isMovementStarted = false;

    void Start()
    {
        ApplyDifficultySettings();
        targetRadius = radius; // 初始目标半径
        
        // 订阅阶段变化事件
        if (stageFlowManager != null)
        {
            stageFlowManager.OnStageChanged += OnStageChanged;
        }
        
        // 初始化位置但不开始运动
        InitializePosition();
    }

    void Update()
    {
        // 只有在收到nAnM阶段信号后才开始运动
        if (!isMovementStarted)
            return;

        // 累计时间
        timeElapsed += Time.deltaTime;

        // 只有在启用半径变化且没有正在进行半径变化时，才响应空格键
        if (enableRadiusChange && Input.GetKeyDown(KeyCode.Space))
        {
            TriggerRadiusChange(3f); // 在 3 秒内进行半径变化
        }

        // 计算新的角度
        angle -= currentSpeed * Time.deltaTime;
        angle = (angle % 360f + 360f) % 360f;
        speedChangePhase -= currentDifficulty_setting.initialSpeed * Time.deltaTime;
        speedChangePhase = (speedChangePhase % 360f) % 360f;

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

        // 基于实际时间改变角速度，而不是基于角度
        currentSpeed = currentDifficulty_setting.initialSpeed + Mathf.Cos(Mathf.Deg2Rad * speedChangeFrequency * speedChangePhase) * currentDifficulty_setting.speedChangeRate;

        if (secondaryObject != null && secondaryObject.activeSelf)
        {
            secX = Mathf.Cos(Mathf.Deg2Rad * angle) * secondaryRadius;
            secY = Mathf.Sin(Mathf.Deg2Rad * angle) * secondaryRadius;
            secondaryObject.transform.position = new Vector3(secX, secY, secondaryObject.transform.position.z);
        }
    }

    // 初始化位置但不开始运动
    private void InitializePosition()
    {
        // 设置初始位置（角度为0时的位置）
        float x = radius;
        float y = 0f;
        transform.position = new Vector3(x, y, transform.position.z);
        
        // 设置副物体的初始位置
        if (secondaryObject != null)
        {
            secX = secondaryRadius;
            secY = 0f;
            secondaryObject.transform.position = new Vector3(secX, secY, secondaryObject.transform.position.z);
        }
        
        // 启动副物体闪烁协程
        secondaryObjectBlinkCoroutine = StartCoroutine(SecondaryObjectBlink());
        
        // 启动半径变化协程（但是否实际触发变化取决于enableRadiusChange）
        radiusChangeCoroutine = StartCoroutine(AutoRadiusChange());
    }

    // 阶段变化事件处理
    private void OnStageChanged(string stageName)
    {
        if (stageName == "nAnM" && !isMovementStarted)
        {
            isMovementStarted = true;
            Debug.Log("收到nAnM阶段信号，开始运动");
        }
    }

    // 根据难度级别应用设置
    public void SetDifficulty(string difficulty)
    {
        if (currentDifficulty != difficulty)
        {
            currentDifficulty = difficulty;
            ApplyDifficultySettings();
            Debug.Log($"难度已切换到: {difficulty}");
        }
    }

    // 应用当前难度设置
    private void ApplyDifficultySettings()
    {
        switch (currentDifficulty.ToLower())
        {
            case "low":
                currentDifficulty_setting = easyDifficulty;
                break;
            case "medium":
                currentDifficulty_setting = mediumDifficulty;
                break;
            case "high":
                currentDifficulty_setting = hardDifficulty;
                break;
            default:
                // 默认使用中等难度
                currentDifficulty_setting = mediumDifficulty;
                break;
        }
    }

    public void SetFeedbackLevel(string feedback)
    {
        switch (feedback.ToLower())
        {
            case "low":
                appearInterval = 0.1f;
                disappearDuration = 0.9f;
                break;
            case "medium":
                appearInterval = 0.4f;
                disappearDuration = 0.6f;
                break;
            case "high":
                appearInterval = 0.8f;
                disappearDuration = 0.2f;
                break;
        }
    }

    public void TriggerRadiusChange(float T)
    {
        // 只有在启用半径变化且没有正在进行半径变化时才触发
        if (enableRadiusChange && !isRadiusChanging)
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
        // 记录上次状态改变的时间
        float lastStateChangeTime = Time.time;
        bool isObjectVisible = false;
        float currentInterval = disappearDuration; // 初始为消失时间

        while (true)
        {
            if (secondaryObject != null)
            {
                float currentTime = Time.time;
                float elapsedTime = currentTime - lastStateChangeTime;

                // 检查是否应该切换状态
                if (isObjectVisible && elapsedTime >= appearInterval)
                {
                    // 从显示切换到隐藏
                    secondaryObject.SetActive(false);
                    isObjectVisible = false;
                    lastStateChangeTime = currentTime;
                    currentInterval = disappearDuration;
                }
                else if (!isObjectVisible && elapsedTime >= disappearDuration)
                {
                    // 从隐藏切换到显示
                    secondaryObject.SetActive(true);
                    isObjectVisible = true;
                    lastStateChangeTime = currentTime;
                    currentInterval = appearInterval;
                }
            }

            yield return null; // 每帧检查一次
        }
    }

    private IEnumerator AutoRadiusChange()
    {
        while (true)
        {
            // 即使未开始运动，也等待指定时间间隔
            yield return new WaitForSeconds(RChangeInterval);
            
            // 只有在启用半径变化、开始运动后才触发半径变化
            if (enableRadiusChange && isMovementStarted)
            {
                TriggerRadiusChange(3f); // 保持3秒的过渡时间
            }
        }
    }

    // 在对象销毁时取消事件订阅
    private void OnDestroy()
    {
        if (stageFlowManager != null)
        {
            stageFlowManager.OnStageChanged -= OnStageChanged;
        }
    }
}