using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class MainMenuController : MonoBehaviour
{
    public GameObject MainMenu;
    public GameObject SettingMenu;
    public GameObject GameUI;
    public GameObject finishUI;
    public StageFlowManager stageFlowManager;

    private void Start()
    {
        // 注册UI面板到UIManager
        if (UIManager.Instance != null)
        {
            UIManager.Instance.RegisterUIPanel("MainMenu", MainMenu);
            UIManager.Instance.RegisterUIPanel("SettingMenu", SettingMenu);
            UIManager.Instance.RegisterUIPanel("FinishUI", finishUI);
            UIManager.Instance.RegisterUIPanel("GameUI", GameUI);
            
            // 显示主菜单
            UIManager.Instance.ShowOnlyUIPanel("MainMenu");
        }
        // 订阅阶段变化事件
        if (stageFlowManager != null)
        {
            stageFlowManager.OnStageChanged += OnStageChanged;
        }
    }

    public void OnPlayButtonClicked()
    {
        if (UIManager.Instance != null)
        {
            UIManager.Instance.ShowOnlyUIPanel("GameUI");
        }
        if (stageFlowManager != null)
        {
            stageFlowManager.StartStageFlow();
        }
    }

    public void OnSettingButtonClicked()
    {
        if (UIManager.Instance != null)
        {
            UIManager.Instance.ShowOnlyUIPanel("SettingMenu");
        }
    }


    public void OnMainMenuButtonClicked()
    {
        if (UIManager.Instance != null)
        {
            UIManager.Instance.ShowUIPanel("MainMenu");
        }
    }

    public void OnExitButtonClicked()
    {
        // 在编辑器中停止播放模式
        #if UNITY_EDITOR
        EditorApplication.isPlaying = false;
        Debug.Log("Editor play mode stopped"); // 添加调试信息
        #else
        // 在构建的应用中退出程序
        Application.Quit();
        Debug.Log("Application quit request sent"); // 添加调试信息
        #endif
    }

    public void OnBackButtonClicked()
    {
        if (UIManager.Instance != null)
        {
            UIManager.Instance.ShowOnlyUIPanel("MainMenu");
        }
    }

    private void OnStageChanged(string stageName)
    {
        if (stageName == "FINISH")
        {
            // 通过UIManager显示结束UI
            if (UIManager.Instance != null)
            {
                UIManager.Instance.ShowUIPanel("FinishUI");
            }

            // 通过BlurManager启用模糊效果
            if (BlurManager.Instance != null)
            {
                BlurManager.Instance.EnableBlur();
            }
        }
    }
}