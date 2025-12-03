using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

[System.Serializable]
public class InequalityData
{
    public int index;
    public string inequality;
    public bool answer;
    public string audio_file;
}

public class AudioTaskController : MonoBehaviour
{
    [Header("听觉任务设置")]
    public bool enableAudioTask = true; // 是否启用听觉任务
    public float audioInterval = 5f; // 每隔5秒播放一次
    public StageFlowManager stageFlowManager; // 在 Inspector 中赋值
    public AudioSource audioSource; // 在 Inspector 中赋值

    private List<InequalityData> inequalityDataList = new List<InequalityData>();
    private AudioClip[] inequalityAudios;

    private int correctCount = 0;
    private int totalCount = 0;
    private bool currentAnswer = false;
    private bool isKeyPressed = false;
    private Coroutine audioTaskCoroutine;

    void Start()
    {
        LoadAndParseCSV();
        LoadAudioClips();

        if (stageFlowManager != null)
        {
            stageFlowManager.OnStageChanged += OnStageChanged;
        }
    }

    private void OnStageChanged(string stageName)
    {
        if (enableAudioTask && (stageName == "aAnM" || stageName == "aAaM"))
        {
            if (audioTaskCoroutine != null)
                StopCoroutine(audioTaskCoroutine);
            audioTaskCoroutine = StartCoroutine(StartAudioTask());
        }
        else if (audioTaskCoroutine != null)
        {
            StopCoroutine(audioTaskCoroutine);
            audioTaskCoroutine = null;
        }
    }

    private IEnumerator StartAudioTask()
    {
        while (true)
        {
            // yield return new WaitForSeconds(audioInterval);

            int randomIndex = Random.Range(0, inequalityAudios.Length);
            currentAnswer = GetAnswerByIndex(randomIndex);

            Debug.Log($"播放不等式 {randomIndex}，答案是 {currentAnswer}");

            if (inequalityAudios[randomIndex] != null && audioSource != null)
            {
                audioSource.PlayOneShot(inequalityAudios[randomIndex]);
            }

            isKeyPressed = false;
            float timer = 0f;

            while (timer < audioInterval)
            {
                if (Input.GetKeyDown(KeyCode.F))
                {
                    isKeyPressed = true;
                    break;
                }

                timer += Time.deltaTime;
                yield return null;
            }

            if (isKeyPressed && !currentAnswer)
            {
                correctCount++;
            }
            else if (!isKeyPressed && currentAnswer)
            {
                correctCount++;
            }

            totalCount++;

            Debug.Log($"当前准确率: {(totalCount > 0 ? (float)correctCount / totalCount * 100 : 0):F2}%");
        }
    }

    private void LoadAndParseCSV()
    {
        string filePath = Path.Combine(Application.dataPath, "inequality_audios", "inequalities.csv");
        if (!File.Exists(filePath))
        {
            Debug.LogError($"找不到CSV文件: {filePath}");
            return;
        }

        string[] lines = File.ReadAllLines(filePath);
        inequalityDataList.Clear();

        for (int i = 1; i < lines.Length; i++)
        {
            string[] fields = lines[i].Split(',');

            if (fields.Length >= 4)
            {
                InequalityData data = new InequalityData
                {
                    index = int.Parse(fields[0]),
                    inequality = fields[1],
                    answer = fields[2].ToLower() == "true",
                    audio_file = fields[3]
                };

                inequalityDataList.Add(data);
            }
        }

        Debug.Log($"成功加载 {inequalityDataList.Count} 条不等式数据");
    }

    private void LoadAudioClips()
    {
        if (inequalityDataList.Count == 0)
        {
            Debug.LogWarning("未加载到不等式数据，无法加载音频");
            return;
        }

        inequalityAudios = new AudioClip[inequalityDataList.Count];

        for (int i = 0; i < inequalityDataList.Count; i++)
        {
            string audioPath = "inequality_audios/" + Path.GetFileNameWithoutExtension(inequalityDataList[i].audio_file);
            AudioClip clip = Resources.Load<AudioClip>(audioPath);

            if (clip != null)
            {
                inequalityAudios[i] = clip;
            }
            else
            {
                Debug.LogWarning($"未找到音频文件: {audioPath}");
            }
        }
    }

    public bool GetAnswerByIndex(int index)
    {
        if (index >= 0 && index < inequalityDataList.Count)
        {
            return inequalityDataList[index].answer;
        }
        return false;
    }

    void OnGUI()
    {
        if (enableAudioTask && totalCount > 0)
        {
            float accuracy = (float)correctCount / totalCount * 100f;
            GUI.Label(new Rect(10, 50, 300, 20), $"听觉任务准确率: {accuracy:F2}% ({correctCount}/{totalCount})");
        }
    }

    void OnDestroy()
    {
        if (stageFlowManager != null)
        {
            stageFlowManager.OnStageChanged -= OnStageChanged;
        }
    }
}