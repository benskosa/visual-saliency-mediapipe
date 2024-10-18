using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BandwidthTester : DataProcessor
{
    public DebugServer debugServer = null;
    int numBytes = 0;
    float startTime = 0;
    float printInterval = 3;

    public override void ProcessData(RecognitionData data, uint width, uint height)
    {
        // not using this function
    }

    // this may not be called from the main thread, so just add the data size
    // to the total number of bytes, and calculate the bandwidth in Update()
    public override void ProcessData(byte[] data, uint width, uint height)
    {
        numBytes += data.Length;
    }

    void Start()
    {
        startTime = Time.time;
    }

    void Update()
    {
        if (Time.time - startTime > printInterval)
        {
            debugServer.SendDebug("Bandwidth: " + numBytes / 1024.0 / (Time.time - startTime) + " KB/s");
            numBytes = 0;
            startTime = Time.time;
        }
    }
}
