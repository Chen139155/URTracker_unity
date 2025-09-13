using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class FinishUIController : MonoBehaviour
{
    public StageFlowManager stageFlowManager; // 在Inspector中拖入StageFlowManager
    [Header("RawImage with Blur Material")]
    public RawImage blurImage;

    [Header("UI Panel to Show/Hide")]
    public GameObject mainUIPanel;
    public GameObject finishUI;

    private Material blurMaterial;

    public Camera mainCamera;       // 主摄像机
    public RenderTexture blurRT;    // 高斯模糊RenderTexture
    public float blurDelay = 0.1f;  // 延迟时间

    void Start()
    {
        
        // 订阅阶段变化事件
        if (stageFlowManager != null)
        {
            stageFlowManager.OnStageChanged += OnStageChanged;
        }
        // 获取材质实例，避免修改共享材质
        if (blurImage != null && blurImage.material != null)
        {
            blurMaterial = Instantiate(blurImage.material);
            blurImage.material = blurMaterial;
        }
        

        
    }

    

    // Button 点击事件调用
    public void OnStageChanged(string stageName)
    {
        if(stageName == "FINISH"){
            // 隐藏game UI
            if (mainUIPanel != null)
                mainUIPanel.SetActive(false);
            if (finishUI != null)
                finishUI.SetActive(true);
            // 打开 RawImage 显示模糊
            if (blurImage != null)
            {
                blurImage.texture = blurRT;
                blurImage.gameObject.SetActive(true);
            }

            // 摄像机输出切换到 RenderTexture
            if (mainCamera != null)
            {
                mainCamera.targetTexture = blurRT;
            }
        }
    }
    // 可选：动态调整模糊强度
    public void SetBlur(float value)
    {
        if (blurMaterial != null)
            blurMaterial.SetFloat("_Size", value);
    }
}
