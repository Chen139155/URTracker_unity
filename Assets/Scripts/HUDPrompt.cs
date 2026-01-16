using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class HUDPrompt : MonoBehaviour
{
    [Header("目标设置")]
    public Transform target; // 要跟随的目标物体
    public Transform secondaryTarget; // 第二个目标物体（如果需要）
    
    [Header("三角形设置")]
    public float triangleSize = 100f; // 三角形初始大小
    public Color triangleColor = Color.green; // 三角形颜色
    public float animationDuration = 1.0f; // 动画持续时间
    public float rotationSpeed = 360f; // 旋转速度（度/秒）
    
    [Header("UI设置")]
    public Canvas canvas; // HUD Canvas
    public Camera mainCamera; // 主摄像机
    
    private RectTransform triangleRect;
    private Image triangleImage;
    private Vector2 targetScreenPos;
    private bool isAnimating = false;
    
    void Start()
    {
        // 创建三角形UI元素
        CreateTriangleUI();
    }
    
    void Update()
    {
        if (target == null) return;
        
        // 将3D目标位置转换为2D屏幕坐标
        Vector3 worldPos = target.position;
        targetScreenPos = mainCamera.WorldToScreenPoint(worldPos);
        
        // 根据Canvas渲染模式进行不同的坐标转换
        if (canvas.renderMode == RenderMode.ScreenSpaceCamera)
        {
            // 对于Screen Space - Camera模式，需要转换为Canvas的局部坐标
            Vector2 localPos;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                canvas.GetComponent<RectTransform>(),
                targetScreenPos,
                canvas.worldCamera,
                out localPos
            );
            triangleRect.localPosition = localPos;
        }
        else if (canvas.renderMode == RenderMode.ScreenSpaceOverlay)
        {
            // 对于Screen Space - Overlay模式，需要将屏幕坐标转换为Canvas的局部坐标
            Vector2 localPos;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                canvas.GetComponent<RectTransform>(),
                targetScreenPos,
                null, // ScreenSpaceOverlay模式下不需要相机
                out localPos
            );
            triangleRect.localPosition = localPos;
        }
        else if (canvas.renderMode == RenderMode.WorldSpace)
        {
            // 对于World Space模式，需要将屏幕坐标转换为世界坐标
            Vector3 worldPosOnCanvas;
            if (RectTransformUtility.ScreenPointToWorldPointInRectangle(
                canvas.GetComponent<RectTransform>(),
                targetScreenPos,
                mainCamera,
                out worldPosOnCanvas
            ))
            {
                triangleRect.position = worldPosOnCanvas;
            }
        }
        
        // 添加调试信息
        Debug.Log($"Target World Pos: {worldPos}");
        Debug.Log($"Target Screen Pos: {targetScreenPos}");
        Debug.Log($"Canvas Render Mode: {canvas.renderMode}");
        Debug.Log($"Triangle Rect Position: {triangleRect.position}");
        
        // 如果不在动画中，开始动画
        if (!isAnimating)
        {
            StartCoroutine(RotateScaleAnimation());
        }
    }
    
    private void CreateTriangleUI()
    {
        // 创建三角形游戏对象
        GameObject triangleGO = new GameObject("HUDPromptTriangle");
        triangleGO.transform.SetParent(canvas.transform, false);
        
        // 添加RectTransform组件
        triangleRect = triangleGO.AddComponent<RectTransform>();
        triangleRect.sizeDelta = new Vector2(triangleSize, triangleSize);
        triangleRect.anchorMin = new Vector2(0.5f, 0.5f);
        triangleRect.anchorMax = new Vector2(0.5f, 0.5f);
        triangleRect.pivot = new Vector2(0.5f, 0.5f);
        
        // 添加Image组件
        triangleImage = triangleGO.AddComponent<Image>();
        triangleImage.color = triangleColor;
        triangleImage.sprite = CreateTriangleSprite();
        triangleImage.raycastTarget = false; // 防止影响其他UI交互
    }
    
    private Sprite CreateTriangleSprite()
    {
        // 创建一个简单的三角形精灵
        Texture2D texture = new Texture2D(100, 100);
        Color[] colors = new Color[100 * 100];
        
        // 填充透明色
        for (int i = 0; i < colors.Length; i++)
        {
            colors[i] = Color.clear;
        }
        
        // 绘制三角形
        for (int y = 0; y < 100; y++)
        {
            for (int x = 50 - y / 2; x <= 50 + y / 2; x++)
            {
                if (x >= 0 && x < 100)
                {
                    colors[y * 100 + x] = Color.white;
                }
            }
        }
        
        texture.SetPixels(colors);
        texture.Apply();
        
        // 创建精灵
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, 100, 100), new Vector2(0.5f, 0.5f));
        return sprite;
    }
    
    private IEnumerator RotateScaleAnimation()
    {
        isAnimating = true;
        
        float elapsedTime = 0f;
        Quaternion startRotation = triangleRect.rotation;
        Quaternion endRotation = startRotation * Quaternion.Euler(0, 0, rotationSpeed * animationDuration);
        
        Vector3 startScale = new Vector3(2f, 2f, 1f);
        Vector3 endScale = new Vector3(1f, 1f, 1f);
        
        // 设置初始状态
        triangleRect.localScale = startScale;
        
        while (elapsedTime < animationDuration)
        {
            // 插值计算旋转角度
            triangleRect.rotation = Quaternion.Lerp(startRotation, endRotation, elapsedTime / animationDuration);
            
            // 插值计算缩放比例
            triangleRect.localScale = Vector3.Lerp(startScale, endScale, elapsedTime / animationDuration);
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // 确保动画结束时处于正确状态
        triangleRect.rotation = endRotation;
        triangleRect.localScale = endScale;
        
        isAnimating = false;
    }
    
    // 设置目标物体
    public void SetTarget(Transform newTarget)
    {
        target = newTarget;
    }
    
    // 设置第二个目标物体
    public void SetSecondaryTarget(Transform newSecondaryTarget)
    {
        secondaryTarget = newSecondaryTarget;
    }
    
    // 显示提示
    public void ShowPrompt()
    {
        gameObject.SetActive(true);
    }
    
    // 隐藏提示
    public void HidePrompt()
    {
        gameObject.SetActive(false);
    }
}