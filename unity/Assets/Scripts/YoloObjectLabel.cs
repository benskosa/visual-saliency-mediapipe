using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;

public class YoloObjectLabel : RecognitionVisualization
{
    TMP_Text labelText;
    Camera mainCamera = null;
    public bool hasArrow = false;
    GameObject arrow;
    Renderer arrowRend;

    // background color: default to white
    public Color backgroundColor = new Color(1, 1, 1, 1);
    // text color: default to black
    public Color textColor = new Color(0, 0, 0, 1);
    Renderer rend;
    Vector3 originalScale;

    void Awake()
    {
        mainCamera = Camera.main;

        rend = GetComponent<Renderer>();
        labelText = GetComponentInChildren<TMP_Text>();

        rend.material.color = backgroundColor;
        labelText.color = textColor;

        if (hasArrow)
        {
            arrow = transform.Find("Arrow").gameObject;
            arrowRend = arrow.GetComponent<Renderer>();
            arrowRend.material.color = backgroundColor;
        }

        originalScale = transform.localScale;
    }

    void UpdateColor()
    {
        rend.material.color = backgroundColor;
        labelText.color = textColor;

        if (hasArrow)
        {
            arrowRend.material.color = backgroundColor;
        }
    }

    void FaceUser()
    {
        if (mainCamera != null)
        {
            // turn to face the camera
            Vector3 targetPosition = mainCamera.transform.position;
            targetPosition.y = transform.position.y;
            transform.LookAt(targetPosition);
            transform.Rotate(0, 180, 0);
        }
    }

    // Update is called once per frame
    void Update()
    {
        FaceUser();
    }

    // inherit from ProcessRecognitionResult
    public override void ProcessRecognitionResult(
        Vector3 position,
        float distance,
        string class_name,
        Vector2 center,
        uint width,
        uint height,
        Color color,
        RecognitionData data,
        int index
    )
    {
        transform.position = position;
        FaceUser();
        
        labelText.text = class_name;

        Vector3 offset = new Vector3(0, 0.08f, 0);

        // if hitDistance>1, scale the label by sqrt(hitDistance)
        // for hitDistance<1, scale the label by 1
        float scale = Mathf.Sqrt(distance) * 2;
        scale = Math.Max(scale, 1.0f);
        transform.localScale = originalScale * scale;
        offset *= scale;

        backgroundColor = color;

        if (data.LabelColors.Count > index)
        {
            var labelColor = data.LabelColors[index];
            if (labelColor != null)
            {
                var tColor = labelColor;
                var alpha = tColor.A < 5? 1.0f : (float)tColor.A / 100.0f;
                textColor = new Color(tColor.R / 255.0f, tColor.G / 255.0f, tColor.B / 255.0f, alpha);
            }
        }

        if (data.LabelProperty != null)
        {
            var labelProperty = data.LabelProperty;

            if (labelProperty.LabelSize > 0.0f) {
                transform.localScale = originalScale * scale * labelProperty.LabelSize;
                offset *= labelProperty.LabelSize;
            }

            if (labelProperty.TextSize > 0) {
                labelText.fontSize = labelProperty.TextSize;
            }
        }

        if (hasArrow)
        {
            // the entire game object should move up by 0.16 times the scale
            transform.position += offset;
        }

        UpdateColor();
    }
}
