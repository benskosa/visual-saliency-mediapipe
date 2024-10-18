using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using System;

public class IPCSkeleton : MonoBehaviour
{
    [SerializeField] List<DataProcessor> dataProcessors = new List<DataProcessor>();
    uint cameraWidth = 640;
    uint cameraHeight = 360;
    uint cameraFps = 30;
    public LocatableCamera locatableCamera = null;
    // accept a function that takes in a float
    // using Unity Event System
    [SerializeField] DisplayCameraStats displayCameraStats;


    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        GetMessage(); // Process next message (if any)
    }

    // Get next message and process it
    bool GetMessage()
    {
        uint command;
        byte[] data;
        if (!hl2ss.PullMessage(out command, out data)) { return false; } // If there are no messages in the queue return false else get the next message

        // Yuheng @ 9/5/2024: Simply process the message, no need to send back the result
        ProcessMessage(command, data);
        // hl2ss.PushResult(ProcessMessage(command, data)); // Process the message and send the result (uint) to the client
        // hl2ss.AcknowledgeMessage(command); // Signal to the library that the message has been processed
        return true; // Done
    }

    // Process message
    uint ProcessMessage(uint command, byte[] data)
    {
        uint ret = 0;

        switch (command)
        {
            // Add your custom message calls here ---------------------------------
            case 0xFFFFFFFCU: ret = MSG_ProcessBackendStats(data); break;
            case 0xFFFFFFFDU: ret = MSG_SetMetaData(data); break;
            case 0xFFFFFFFEU: ret = MSG_ProcessMSG(data); break; // Sample message, feel free to remove it (and its method)
            case 0xFFFFFFFFU: ret = MSG_Disconnect(data); break; // Reserved, do not use 0xFFFFFFFF for your custom messages
        }

        return ret;
    }

    uint MSG_ProcessBackendStats(byte[] data)
    {
        if (!displayCameraStats) { return 0; }
        // data is a json with "framerate" and "bandwidth" keys
        var jsonString = System.Text.Encoding.UTF8.GetString(data);
        JObject json = JObject.Parse(jsonString);
        var framerate = (float)json["framerate"];
        var bandwidth = (float)json["bandwidth"];
        displayCameraStats.SetBackendStats(framerate, bandwidth);
        return 1; // Return 1 to the client to indicate success
    }

    uint MSG_SetMetaData(byte[] data)
    {
        // parse the data
        JObject json = JObject.Parse(System.Text.Encoding.UTF8.GetString(data));

        // first, set the debug server ip address
        var debugServerAddress = (string)json["ip"];
        DebugServer.SetAddress(debugServerAddress);

        // then, set the camera resolution
        cameraWidth = (uint)json["width"];
        cameraHeight = (uint)json["height"];
        cameraFps = (uint)json["fps"];
        DebugServer.SendDebugMessage("Camera resolution set to " + cameraWidth + "x" + cameraHeight + "@" + cameraFps + " fps");

        // then, get the projection matrix (float 1x3)
        var projectionMatrix = (JArray)json["projection"];
        float[] projection = new float[3];
        for (int i = 0; i < 3; i++)
        {
            projection[i] = (float)projectionMatrix[i];
        }
        locatableCamera?.SetProjection(projection);
        DebugServer.SendDebugMessage("Projection matrix set to [" + projection[0] + ", " + projection[1] + ", " + projection[2] + "]");

        return 1; // Return 1 to the client to indicate success
    }

    // Client disconnected
    uint MSG_Disconnect(byte[] data)
    {
        // Implement your OnClientDisconnected logic here
        return ~0U; // Return value does not matter since there is no client anymore
    }

    // Add your custom message methods here -----------------------------------
    // You can use the BitConverter class to unpack data (such as floats, ints) from the data byte array
    // See the RemoteUnityScene.cs script for more examples

    uint MSG_ProcessMSG(byte[] data)
    {
        // TODO: process the message with the DataProcessor objects
        foreach (DataProcessor dp in dataProcessors)
        {
            try {
                dp.ProcessData(data, cameraWidth, cameraHeight);
            } catch (System.Exception e) {
                DebugServer.SendDebugMessage("Error processing data: " + e.Message);
            }
        }
        return 1; // Return 1 to the client to indicate success
    }
}
