using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(RawImage))]
public class UIBlurEffect : MonoBehaviour
{
    public Camera sceneCamera; // 渲染背景的摄像机
    public int downsample = 2; // 降采样倍数，提高性能
    public float blurSize = 2f;

    private RenderTexture rt;
    private RawImage rawImage;
    private Material blurMaterial;

    void Awake()
    {
        rawImage = GetComponent<RawImage>();
        blurMaterial = new Material(Shader.Find("Custom/UIBlur"));
        rawImage.material = blurMaterial;
    }

    void OnEnable()
    {
        SetupRenderTexture();
    }

    void OnDisable()
    {
        if (rt != null)
        {
            rt.Release();
            rt = null;
        }
    }

    void SetupRenderTexture()
    {
        if (sceneCamera == null)
        {
            Debug.LogError("请指定场景摄像机！");
            return;
        }

        int width = Screen.width / downsample;
        int height = Screen.height / downsample;

        rt = new RenderTexture(width, height, 16);
        sceneCamera.targetTexture = rt;

        rawImage.texture = rt;
    }

    void Update()
    {
        if (blurMaterial != null)
        {
            blurMaterial.SetFloat("_BlurSize", blurSize);
        }
    }
}
