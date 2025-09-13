using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class BlurController : MonoBehaviour
{
    [Header("RawImage with Blur Material")]
    public RawImage blurImage;

    [Header("UI Panel to Show/Hide")]
    public GameObject mainUIPanel;

    private Material blurMaterial;

    public Camera mainCamera;       // 主摄像机
    public RenderTexture blurRT;    // 高斯模糊RenderTexture
    public float blurDelay = 0.1f;  // 延迟时间

    void Start()
    {
        // 开场先渲染到屏幕，避免 No Cameras Rendering
        mainCamera.targetTexture = null;
        mainCamera.enabled = true;

        // RawImage 先隐藏
        if (blurImage != null)
            blurImage.gameObject.SetActive(false);

        // 获取材质实例，避免修改共享材质
        if (blurImage != null && blurImage.material != null)
        {
            blurMaterial = Instantiate(blurImage.material);
            blurImage.material = blurMaterial;
        }

        // 启动协程，延迟显示模糊
        StartCoroutine(ShowBlurAfterDelay());
    }

    IEnumerator ShowBlurAfterDelay()
    {
        yield return new WaitForSeconds(blurDelay);

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

    // Button 点击事件调用
    public void OnStartButtonClicked()
    {
        // 隐藏启动 UI
        if (mainUIPanel != null)
            mainUIPanel.SetActive(false);

        // 关闭模糊
        if (blurMaterial != null)
            blurMaterial.SetFloat("_Size", 0);

        if (mainCamera != null)
        {
            // 将摄像机输出重置回屏幕
            mainCamera.targetTexture = null;

            // 可选：释放RenderTexture节省显存
            if (blurRT != null)
                blurRT.Release();
        }
    }

    // 可选：动态调整模糊强度
    public void SetBlur(float value)
    {
        if (blurMaterial != null)
            blurMaterial.SetFloat("_Size", value);
    }
}
