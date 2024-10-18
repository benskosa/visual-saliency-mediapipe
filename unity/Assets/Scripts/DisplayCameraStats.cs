using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using TMPro;
using System.Runtime.InteropServices;
using AOT;

public class DisplayCameraStats : MonoBehaviour
{
    static DisplayCameraStats _instance = null;
    [SerializeField] TMP_Text receiveText;
    [SerializeField] TMP_Text sendText;
    [SerializeField] TMP_Text backendReceiveText;
    int cameraReceiveCnt = 0;
    int cameraSendCnt = 0;
    float cameraReceiveLastReportTime = 0.0f; // in seconds
    float cameraSendLastReportTime = 0.0f; // in seconds
    float currentTime = 0.0f; // in seconds
    uint totalBytesSent = 0;
    float backendFramerate = 0.0f;
    float backendBandwidth = 0.0f;
    bool backendShouldUpdate = false;
    float backendLastReportTime = 0.0f; // in seconds
    long systemTotalDelay = 0;
    int systemTotalDelayCnt = 0;
    long lastSystemDelay = 0;
    long networkTotalDelay = 0;
    int networkTotalDelayCnt = 0;
    long lastNetworkDelay = 0;

    void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        try {
            hl2ss.AddCustomFrameCallback(UpdateCameraReceiveCallback);
            hl2ss.AddCustomFrameSentCallback(UpdateCameraSendCallback);

            // set initial time
            cameraReceiveLastReportTime = Time.time;
            cameraSendLastReportTime = Time.time;

            // set debug callback
            hl2ss.SetUnityDebug(UnityDebugCallback);

            // set delay callback
            hl2ss.SetNetworkDelayCallback(SetNetworkDelayCallback);
            hl2ss.SetSystemDelayCallback(SetSystemDelayCallback);

        }
        catch (Exception e) {
            DebugServer.SendDebugMessage("Can't register callback: " + e.Message);
        }
    }

    void Update()
    {
        currentTime = Time.time;

        if (currentTime - cameraReceiveLastReportTime > 1.0f)
        {
            double receiveFPS = cameraReceiveCnt / (currentTime - cameraReceiveLastReportTime);
            // keep three decimal places
            receiveText.text = "Acquire:\n" + receiveFPS.ToString("F3") + " FPS";
            cameraReceiveCnt = 0;
            cameraReceiveLastReportTime = currentTime;
        }

        if (currentTime - cameraSendLastReportTime > 1.0f)
        {
            double sendFPS = cameraSendCnt / (currentTime - cameraSendLastReportTime);
            double bandwidth = totalBytesSent * 8 / 1000 / (currentTime - cameraSendLastReportTime); // in Kbps
            // keep three decimal places
            sendText.text = "Send:\n" + sendFPS.ToString("F3") + " FPS\n" + bandwidth.ToString("F3") + " Kbps";
            cameraSendCnt = 0;
            totalBytesSent = 0;
            cameraSendLastReportTime = currentTime;
        }

        if (backendShouldUpdate || currentTime - backendLastReportTime > 1.0f)
        {
            UpdateBackendText();
            backendShouldUpdate = false;
            backendLastReportTime = currentTime;
        }
    }

    void UpdateCameraReceive(ulong timestamp, float[] pose)
    {
        cameraReceiveCnt++;
    }

    void UpdateCameraSend(uint bytesSent)
    {
        cameraSendCnt++;
        totalBytesSent += bytesSent;
    }

    public void SetBackendStats(float framerate, float bandwidth)
    {
        backendFramerate = framerate;
        backendBandwidth = bandwidth; // not used
    }
    
    void UpdateBackendText()
    {
        long systemDelayToDisplay = systemTotalDelayCnt == 0? lastSystemDelay : systemTotalDelay / systemTotalDelayCnt;
        long networkDelayToDisplay = networkTotalDelayCnt == 0? lastNetworkDelay : networkTotalDelay / networkTotalDelayCnt;

        backendReceiveText.text = "Receive: " + backendFramerate.ToString("F3") + " FPS\n" + 
            "System Delay: " + systemDelayToDisplay.ToString() + " ms\n" +
            "Network Delay: " + networkDelayToDisplay.ToString() + " ms";

        systemTotalDelay = 0;
        systemTotalDelayCnt = 0;
        lastSystemDelay = systemDelayToDisplay;
        networkTotalDelay = 0;
        networkTotalDelayCnt = 0;
        lastNetworkDelay = networkDelayToDisplay;
    }

    void UpdateSystemDelay(long delay)
    {
        systemTotalDelay += delay;
        systemTotalDelayCnt++;
    }

    void UpdateNetworkDelay(long delay)
    {
        networkTotalDelay += delay;
        networkTotalDelayCnt++;
    }

    // static methods
    [MonoPInvokeCallback(typeof(hl2ss.FrameCallback))]
    public static void UpdateCameraReceiveCallback(ulong timestamp, float[] pose)
    {
        if (_instance != null) {
            try {
                _instance.UpdateCameraReceive(timestamp, pose);
            }
            catch (Exception e) {
                DebugServer.SendDebugMessage("UpdateCameraReceiveCallbackError: " + e.Message);
            }
        }
    }

    [MonoPInvokeCallback(typeof(hl2ss.FrameSentCallback))]
    public static void UpdateCameraSendCallback(uint bytesSent)
    {
        if (_instance != null) {
            try {
                _instance.UpdateCameraSend(bytesSent);
            }
            catch (Exception e) {
                DebugServer.SendDebugMessage("UpdateCameraSendCallbackError: " + e.Message);
            }
        }
    }

    [MonoPInvokeCallback(typeof(hl2ss.UnityDebugCallback))]
    public static void UnityDebugCallback(string message)
    {
        DebugServer.SendDebugMessage(message);
    }

    [MonoPInvokeCallback(typeof(hl2ss.DelayCallback))]
    public static void SetNetworkDelayCallback(long delay)
    {
        if (_instance != null) {
            try {
                _instance.UpdateNetworkDelay(delay);
            }
            catch (Exception e) {
                DebugServer.SendDebugMessage("SetDelayCallbackError: " + e.Message);
            }
        }
    }

    [MonoPInvokeCallback(typeof(hl2ss.DelayCallback))]
    public static void SetSystemDelayCallback(long delay)
    {
        if (_instance != null) {
            try {
                _instance.UpdateSystemDelay(delay);
            }
            catch (Exception e) {
                DebugServer.SendDebugMessage("SetDelayCallbackError: " + e.Message);
            }
        }
    }
}
