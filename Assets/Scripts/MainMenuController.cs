using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MainMenuController : MonoBehaviour
{
    public GameObject MainMenu;
    public GameObject SettingMenu;

    // Start is called before the first frame update
    void Start()
    {
        MainMenuButton();
        MainMenu.SetActive(true);
        SettingMenu.SetActive(false);
    }

    public void PlayButton()
    {
        // Play Now Button has been pressed, here you can initialize your game (For example Load a Scene called GameLevel etc.)
        UnityEngine.SceneManagement.SceneManager.LoadScene("GameScene");
    }

    public void SettingButton()
    {
        // Show Credits Menu
        MainMenu.SetActive(false);
        SettingMenu.SetActive(true);
    }

    public void MainMenuButton()
    {
        // Show Main Menu
        MainMenu.SetActive(true);
        SettingMenu.SetActive(false);
    }

    public void QuitButton()
    {
        // Quit Game
        Application.Quit();
    }
}
