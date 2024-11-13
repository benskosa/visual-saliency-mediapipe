// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using System;

// public class YoloObjectIcon : RecognitionVisualization
// {

//     Camera mainCamera = null;
//     GameObject arrow;
//     Renderer arrowRend;

//     // background color: default to white
//     public Color backgroundColor = new Color(1, 0, 0, 1);
//     // text color: default to black
//     public Color iconColor = new Color(1, 1, 1, 1);
//     public bool hasBackground = true;
//     public bool hasArrow = false;
//     Renderer rend;
//     Vector3 originalScale;
//     GameObject icon;
//     SpriteRenderer spriteRenderer;

//     void Awake()
//     {
//         mainCamera = Camera.main;

//         if (hasBackground)
//             rend = GetComponent<Renderer>();
        
//         if (hasArrow)
//         {
//             arrow = transform.Find("Arrow").gameObject;
//             arrowRend = arrow.GetComponent<Renderer>();
//         }

//         originalScale = transform.localScale;

//         // icon is the child of the object
//         icon = transform.Find("Icon").gameObject;
//         spriteRenderer = icon.GetComponent<SpriteRenderer>();

//         UpdateColor();
//     }

//     void FaceUser()
//     {
//         if (mainCamera != null)
//         {
//             // turn to face the camera
//             Vector3 targetPosition = mainCamera.transform.position;
//             targetPosition.y = transform.position.y;
//             transform.LookAt(targetPosition);
//             transform.Rotate(0, 180, 0);
//         }
//     }

//     void UpdateColor()
//     {
//         if (hasBackground) {
//             rend.material.color = backgroundColor;
//         }

//         spriteRenderer.color = iconColor;

//         if (hasArrow) {
//             if (hasBackground)
//                 arrowRend.material.color = backgroundColor;
//             else
//                 arrowRend.material.color = iconColor;
//         }
//     }

//     // Update is called once per frame
//     void Update()
//     {
//         FaceUser();
//     }

//     // inherit from ProcessRecognitionResult
//     public override void ProcessRecognitionResult(
//         Vector3 position,
//         float distance,
//         string class_name,
//         Vector2 center,
//         uint width,
//         uint height,
//         Color color,
//         RecognitionData data,
//         int index
//     )
//     {
//         transform.position = position;
//         FaceUser();
        
//         var iconSprite = SpriteManager.GetSprite(class_name);
//         if (iconSprite == null)
//         {
//             DebugServer.SendDebugMessage("Icon not found: " + class_name);
//             return;
//         }
//         spriteRenderer.sprite = iconSprite;

//         Vector3 offset = new Vector3(0, 0.08f, 0);

//         // if hitDistance>1, scale the label by sqrt(hitDistance)
//         // for hitDistance<1, scale the label by 1
//         float scale = Mathf.Sqrt(distance) * 2;
//         scale = Math.Max(scale, 1.0f);
//         transform.localScale = originalScale * scale;
//         offset *= scale;

//         backgroundColor = color;

//         if (data.LabelColors.Count > index)
//         {
//             var labelColor = data.LabelColors[index];
//             if (labelColor != null)
//             {
//                 var tColor = labelColor;
//                 var alpha = tColor.A < 5? 1.0f : (float)tColor.A / 100.0f;
//                 iconColor = new Color(tColor.R / 255.0f, tColor.G / 255.0f, tColor.B / 255.0f, alpha);
//             }
//         }

//         if (data.LabelProperty != null)
//         {
//             var labelProperty = data.LabelProperty;

//             if (labelProperty.LabelSize > 0.0f) {
//                 transform.localScale = originalScale * scale * labelProperty.LabelSize;
//                 offset *= labelProperty.LabelSize;
//             }
//         }

//         if (hasArrow)
//         {
//             // the entire game object should move up by 0.16 times the scale
//             transform.position += offset;
//         }

//         UpdateColor();
//     }
// }
