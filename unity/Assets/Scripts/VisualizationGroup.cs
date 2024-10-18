using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VisualizationGroup : RecognitionVisualization
{
    public List<RecognitionVisualization> prefabList;
    List<RecognitionVisualization> instantiatedPrefabs = new List<RecognitionVisualization>();

    void Awake()
    {
        foreach (var prefab in prefabList)
        {
            RecognitionVisualization instance = Instantiate(prefab, transform);
            instantiatedPrefabs.Add(instance);
        }
    }

    void OnDestroy()
    {
        foreach (var instance in instantiatedPrefabs)
        {
            Destroy(instance.gameObject);
        }
    }

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
        foreach (var instance in instantiatedPrefabs)
        {
            instance.ProcessRecognitionResult(
                position,
                distance,
                class_name,
                center,
                width,
                height,
                color,
                data,
                index
            );
        }
    }
}
