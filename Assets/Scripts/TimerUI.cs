using UnityEngine;
using TMPro;   // 需要导入TextMeshPro命名空间

public class TimerUI : MonoBehaviour
{
    public TextMeshProUGUI timerText;  // 拖入UI组件
    private float startTime;
    private bool isTiming = false;

    void Start()
    {
        StartTimer();  // 游戏一开始就计时
    }

    void Update()
    {
        if (isTiming)
        {
            float elapsed = Time.time - startTime;

            // 转换成 分:秒 格式
            int minutes = Mathf.FloorToInt(elapsed / 60f);
            int seconds = Mathf.FloorToInt(elapsed % 60f);

            timerText.text = string.Format("{0:00}:{1:00}", minutes, seconds);
        }
    }

    public void StartTimer()
    {
        startTime = Time.time;
        isTiming = true;
    }

    public void StopTimer()
    {
        isTiming = false;
    }
}
