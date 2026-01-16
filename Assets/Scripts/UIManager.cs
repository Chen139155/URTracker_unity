using UnityEngine;
using System.Collections.Generic;

public class UIManager : MonoBehaviour
{
    public static UIManager Instance { get; private set; }

    // 所有UI面板的字典管理
    private Dictionary<string, GameObject> uiPanels = new Dictionary<string, GameObject>();
    private string currentActivePanel = string.Empty;

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
    }

    /// <summary>
    /// 注册UI面板
    /// </summary>
    public void RegisterUIPanel(string panelName, GameObject panel)
    {
        if (!uiPanels.ContainsKey(panelName))
        {
            uiPanels.Add(panelName, panel);
        }
        else
        {
            Debug.LogWarning($"UI面板 {panelName} 已存在");
        }
    }

    /// <summary>
    /// 显示指定UI面板
    /// </summary>
    public void ShowUIPanel(string panelName)
    {
        if (uiPanels.ContainsKey(panelName))
        {
            // 隐藏当前激活的面板
            if (!string.IsNullOrEmpty(currentActivePanel) && uiPanels.ContainsKey(currentActivePanel))
            {
                uiPanels[currentActivePanel].SetActive(false);
            }

            // 显示新面板
            uiPanels[panelName].SetActive(true);
            currentActivePanel = panelName;
            Debug.Log($"显示UI面板: {panelName}");
        }
        else
        {
            Debug.LogError($"未找到UI面板: {panelName}");
        }
    }

    /// <summary>
    /// 隐藏指定UI面板
    /// </summary>
    public void HideUIPanel(string panelName)
    {
        if (uiPanels.ContainsKey(panelName))
        {
            uiPanels[panelName].SetActive(false);
            Debug.Log($"隐藏UI面板: {panelName}");
        }
    }

    /// <summary>
    /// 显示指定UI面板并隐藏其他所有面板
    /// </summary>
    public void ShowOnlyUIPanel(string panelName)
    {
        foreach (var kvp in uiPanels)
        {
            kvp.Value.SetActive(kvp.Key == panelName);
        }
        currentActivePanel = panelName;
    }

    /// <summary>
    /// 获取当前激活的面板名称
    /// </summary>
    public string GetCurrentActivePanel()
    {
        return currentActivePanel;
    }

    /// <summary>
    /// 检查面板是否处于激活状态
    /// </summary>
    public bool IsPanelActive(string panelName)
    {
        if (uiPanels.ContainsKey(panelName))
        {
            return uiPanels[panelName].activeSelf;
        }
        return false;
    }
}
