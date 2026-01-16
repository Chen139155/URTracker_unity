using UnityEngine;
using UnityEngine.EventSystems;

/// <summary>
/// Button hover effect that scales up the button by 10% when mouse enters
/// </summary>
public class ButtonHoverScale : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
{
    [SerializeField] private float scaleFactor = 1.1f; // Scale up by 10%
    [SerializeField] private float transitionSpeed = 5f; // Smooth transition speed
    
    private Vector3 originalScale; // Store original scale
    private Vector3 targetScale; // Target scale for smooth transition

    private void Start()
    {
        // Save the original scale of the button
        originalScale = transform.localScale;
        targetScale = originalScale;
    }

    private void Update()
    {
        // Smoothly transition to the target scale
        transform.localScale = Vector3.Lerp(transform.localScale, targetScale, Time.deltaTime * transitionSpeed);
    }

    /// <summary>
    /// Called when mouse enters the button area
    /// </summary>
    public void OnPointerEnter(PointerEventData eventData)
    {
        // Set target scale to 110% of original size
        targetScale = originalScale * scaleFactor;
    }

    /// <summary>
    /// Called when mouse exits the button area
    /// </summary>
    public void OnPointerExit(PointerEventData eventData)
    {
        // Set target scale back to original size
        targetScale = originalScale;
    }
}