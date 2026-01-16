using UnityEngine;
using UnityEngine.UI;

public class BlurManager : MonoBehaviour
{
    public static BlurManager Instance { get; private set; }

    [Header("Blur Settings")]
    public RawImage blurImage;
    public Camera mainCamera;
    public RenderTexture blurRT;
    public float blurDelay = 0.1f;

    private Material blurMaterial;
    private bool isBlurActive = false;

    private void Awake()
    {
        // 单例模式实现
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
            return;
        }

        // 初始化模糊材质
        if (blurImage != null && blurImage.material != null)
        {
            blurMaterial = Instantiate(blurImage.material);
            blurImage.material = blurMaterial;
            blurImage.gameObject.SetActive(false);
        }

        // 初始化摄像机
        if (mainCamera != null)
        {
            mainCamera.targetTexture = null;
        }
    }

    /// <summary>
    /// 启用模糊效果
    /// </summary>
    public void EnableBlur()
    {
        if (!isBlurActive && blurImage != null && mainCamera != null)
        {
            // 设置模糊纹理和摄像机目标
            blurImage.texture = blurRT;
            mainCamera.targetTexture = blurRT;
            blurImage.gameObject.SetActive(true);
            isBlurActive = true;
            Debug.Log("启用模糊效果");
        }
    }

    /// <summary>
    /// 禁用模糊效果
    /// </summary>
    public void DisableBlur()
    {
        if (isBlurActive && blurImage != null && mainCamera != null)
        {
            // 重置摄像机目标和隐藏模糊图像
            mainCamera.targetTexture = null;
            blurImage.gameObject.SetActive(false);
            isBlurActive = false;
            Debug.Log("禁用模糊效果");
        }
    }

    /// <summary>
    /// 延迟启用模糊效果
    /// </summary>
    public void EnableBlurWithDelay()
    {
        StartCoroutine(EnableBlurAfterDelay());
    }

    private System.Collections.IEnumerator EnableBlurAfterDelay()
    {
        yield return new WaitForSeconds(blurDelay);
        EnableBlur();
    }

    /// <summary>
    /// 设置模糊强度
    /// </summary>
    public void SetBlurIntensity(float value)
    {
        if (blurMaterial != null)
        {
            blurMaterial.SetFloat("_Size", value);
        }
    }

    /// <summary>
    /// 获取当前模糊状态
    /// </summary>
    public bool IsBlurActive()
    {
        return isBlurActive;
    }
}
