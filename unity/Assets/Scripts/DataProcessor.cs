using System.Collections;
using System.Collections.Generic;
using UnityEngine;
// using Newtonsoft.Json.Linq;
using Google.Protobuf;

public abstract class DataProcessor : MonoBehaviour
{
    public virtual void ProcessData(byte[] data, uint width, uint height) {
        RecognitionData recognitionData = RecognitionData.Parser.ParseFrom(data);
        ProcessData(recognitionData, width, height);
    }

    public abstract void ProcessData(RecognitionData data, uint width, uint height);
}
