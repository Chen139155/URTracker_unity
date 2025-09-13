using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine.UI; // UI 组件
using TMPro;

[System.Serializable]
public class StageInfo
{
    public string stageName;   // 阶段名称
    public float duration;     // 阶段持续时间（秒）
}

public class StageFlowManager : MonoBehaviour
{
    [Header("Stage Flow 配置")]
    public List<StageInfo> stages = new List<StageInfo>();

    [Header("ZMQ 发布管理")]
    public ZmqManager zmqManager; // 在 Inspector 中拖入 ZmqManager

    [Header("UI 显示")]
    public TextMeshProUGUI stageNameText;   // 显示当前 Stage 名称
    public TextMeshProUGUI timerText;       // 显示当前 Stage 的计时（剩余时间或已用时间）

    private Coroutine _stageCoroutine;
    private Coroutine _timerCoroutine; // 专门跑计时的协程
    private int _currentStageIndex = -1;

    // 阶段切换事件，订阅的模块可根据 stageName 修改状态
    public event Action<string> OnStageChanged;

    void Update()
    {
        // 按下 PageDown 键跳过当前阶段
        if (Input.GetKeyDown(KeyCode.PageDown))
        {
            SkipToNextStage();
        }
    }

    /// <summary>
    /// 开始阶段流
    /// </summary>
    public void StartStageFlow()
    {
        if (_stageCoroutine != null)
        {
            StopCoroutine(_stageCoroutine);
        }
        _stageCoroutine = StartCoroutine(RunStageFlow());
    }

    private IEnumerator RunStageFlow()
    {
        for (int i = 0; i < stages.Count; i++)
        {
            _currentStageIndex = i;
            StageInfo stage = stages[i];

            SwitchStage(stage.stageName, stage.duration);

            if (stage.duration > 0)
            {
                yield return new WaitForSeconds(stage.duration);
            }
        }

        Debug.Log("StageFlow 全部完成");
    }

    private void SwitchStage(string stageName, float duration)
    {
        Debug.Log($"切换到阶段: {stageName}");

        // 更新 UI
        if (stageNameText != null)
        {
            stageNameText.text = $"Stage: {stageName}";
        }

        if (timerText != null)
        {
            timerText.text = $"Time: {duration:F2}s";
        }

        // 启动计时协程
        if (_timerCoroutine != null)
        {
            StopCoroutine(_timerCoroutine);
        }
        if (duration > 0 && timerText != null)
        {
            _timerCoroutine = StartCoroutine(UpdateStageTimer(duration));
        }

        // 发布到 ZMQ
        if (zmqManager != null)
        {
            zmqManager.PublishTaskStage(stageName);
        }

        // 触发事件，通知其他模块
        OnStageChanged?.Invoke(stageName);
    }

    private IEnumerator UpdateStageTimer(float duration)
    {
        float timeLeft = duration;
        while (timeLeft > 0f)
        {
            if (timerText != null)
            {
                timerText.text = $"Time: {timeLeft:F2}s";
            }
            yield return null; // 每帧更新
            timeLeft -= Time.deltaTime;
        }
        if (timerText != null)
        {
            timerText.text = "Time: 0.00s";
        }
    }

    /// <summary>
    /// 跳到下一个阶段（可手动控制）
    /// </summary>
    public void SkipToNextStage()
    {
        if (_stageCoroutine != null)
        {
            StopCoroutine(_stageCoroutine);

            if (_currentStageIndex + 1 < stages.Count)
            {
                _stageCoroutine = StartCoroutine(RunStageFlowFrom(_currentStageIndex + 1));
            }
        }
    }

    private IEnumerator RunStageFlowFrom(int startIndex)
    {
        for (int i = startIndex; i < stages.Count; i++)
        {
            _currentStageIndex = i;
            StageInfo stage = stages[i];

            SwitchStage(stage.stageName, stage.duration);

            if (stage.duration > 0)
            {
                yield return new WaitForSeconds(stage.duration);
            }
        }

        Debug.Log("StageFlow 全部完成");
    }
}
