using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// base class for recognition visualization prefab
// every recognition visualization should inherit from this class
public abstract class MediapipeVisualization : MonoBehaviour
{
    // process the mediapipe recognition result
    // including:
    // - position: the position of the object, in Unity world space
    // - distance: the distance of the object from the camera
    // - center: the geometry center of the object
    // - width: the width of the image
    // - height: the height of the image
    // - color: the color of the object
    // - data: the recognition data
    //      Sidenote: this isn't the best way to pass data, but no need to parse the contour if not needed
    //      The other passed fields are either computed by caller or used by caller anyways
    // - index: the index of the object in the recognition result
    public abstract void ProcessRecognitionResult(
        Vector3 position,
        float distance,
        Vector2 center,
        uint width,
        uint height,
        Color color,
        RecognitionData data,
        int index
    );

    // a test function
    public virtual void TestProcessRecognitionResult(
        Vector3 position,
        float distance,
        Vector2 center,
        uint width,
        uint height,
        Color color,
        RecognitionData data,
        int index
    ) {
        // do nothing
    }

    public virtual void ProcessRecognitionResult2D(
        Vector3 position,
        float distance,
        Vector2 center,
        uint width,
        uint height,
        Color color,
        RecognitionData data,
        int index
    ) {
        // by default: process the 2D result as 3D result
        ProcessRecognitionResult(position, distance, center, width, height, color, data, index);
    }
}
