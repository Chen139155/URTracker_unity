using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TatgetControl : MonoBehaviour
{
    // 圆周运动的初始半径
    public float radius = 4f;
    // 初始角速度（角度每秒）
    public float initialSpeed = 40f;
    // 速度变化率（每秒变化的角速度）
    public float speedChangeRate = 30f;
    public float speedChangeFrequency = 3f;
    
    // 当前角度
    private float angle = 0f;
    // 当前角速度
    private float currentSpeed;
    // Start is called before the first frame update
    void Start()
    {
        // 初始化角速度
        currentSpeed = initialSpeed;
    }

    // Update is called once per frame
    void Update()
    {
        // 计算新的角度
        angle -= currentSpeed * Time.deltaTime;

        // 让角度保持在 0 到 360 度之间
        angle = (angle % 360f + 360f) % 360f;

        // 计算 Sprite 的新位置
        float x = Mathf.Cos(Mathf.Deg2Rad * angle) * radius;
        float y = Mathf.Sin(Mathf.Deg2Rad * angle) * radius;

        // 设置 Sprite 的位置，使其围绕屏幕中心旋转
        transform.position = new Vector3(x, y, transform.position.z);

        // 改变角速度，实现变速效果
        currentSpeed = initialSpeed + Mathf.Cos(Mathf.Deg2Rad * angle*speedChangeFrequency) * speedChangeRate;
    }
}
