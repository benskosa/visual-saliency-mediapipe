using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;

[System.Serializable]
public class DummyData
{
    // mask_contours: List[List[List[int]]]
    // Most inner list is a point [x, y]
    public List<List<List<int>>> mask_contours;
    // boxes: List[List[int]]
    // format: [x1, y1, x2, y2]
    public List<List<int>> boxes;
    // scores: List[float]
    public List<float> scores;
    // class_names: List[str]
    public List<string> class_names;
    // geometry_center: List[List[int]]
    // inner list is a point [x, y]
    public List<List<int>> geometry_center;
}

public class DummyParser : MonoBehaviour
{
    public TextAsset dummyJson;
    // Start is called before the first frame update
    void Start()
    {
        if (dummyJson == null)
        {
            Debug.LogError("No dummy json file found");
            return;
        }
        DummyData dummyData = JsonConvert.DeserializeObject<DummyData>(dummyJson.text);
        Debug.Log("Parsed " + dummyData.mask_contours.Count + " objects");

        // do whatever you want with the parsed data
    }
}
