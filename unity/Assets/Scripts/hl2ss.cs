
using System;
using System.Runtime.InteropServices;
using UnityEngine;
using AOT;

public static class hl2ss
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void FrameCallback(ulong timestamp, [MarshalAs(UnmanagedType.LPArray, SizeConst = 16)] float[] pose);
    // this callback takes in a DWORD representing the number of bytes sent
    // DWORD is a 32-bit unsigned integer
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void FrameSentCallback(uint bytesSent);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void UnityDebugCallback(string message);

    // C++ signature: typedef void (*DelayCallback)(const int64_t);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void DelayCallback(long delay);

#if WINDOWS_UWP
    [DllImport("hl2ss")]
    private static extern void InitializeStreamsOnUI(uint enable);
    [DllImport("hl2ss")]
    private static extern void DebugMessage(string str);
    [DllImport("hl2ss")]
    private static extern void MQ_SO_Push(uint value);
    [DllImport("hl2ss")]
    private static extern void MQ_SI_Pop(out uint command, byte[] data);
    [DllImport("hl2ss")]
    private static extern uint MQ_SI_Peek();
    [DllImport("hl2ss")]
    private static extern void MQ_Restart();
    [DllImport("hl2ss")]
    private static extern void GetLocalIPv4Address(byte[] data, int size);
    [DllImport("hl2ss")]
    private static extern int OverrideWorldCoordinateSystem(IntPtr scs);

    [DllImport("hl2ss", CallingConvention = CallingConvention.Cdecl)]
    public static extern int AddCustomFrameCallback(FrameCallback callback);
    [DllImport("hl2ss", CallingConvention = CallingConvention.Cdecl)]
    public static extern int AddCustomFrameSentCallback(FrameSentCallback callback);
    [DllImport("hl2ss", CallingConvention = CallingConvention.Cdecl)]
    public static extern void SetUnityDebug(UnityDebugCallback callback);
    [DllImport("hl2ss", CallingConvention = CallingConvention.Cdecl)]
    public static extern void SetNetworkDelayCallback(DelayCallback callback);
    [DllImport("hl2ss", CallingConvention = CallingConvention.Cdecl)]
    public static extern void SetSystemDelayCallback(DelayCallback callback);


#else
    private static void InitializeStreamsOnUI(uint enable)
    {
    }

    private static void DebugMessage(string str)
    {
        Debug.Log(str);
    }

    private static void MQ_SO_Push(uint value)
    {
    }

    private static void MQ_SI_Pop(out uint command, byte[] data)
    {
        command = ~0U;
    }

    private static uint MQ_SI_Peek()
    {
        return ~0U;
    }

    private static void MQ_Restart()
    {
    }

    private static void GetLocalIPv4Address(byte[] data, int size)
    {
    }

    private static int OverrideWorldCoordinateSystem(IntPtr scs)
    {
        return 1;
    }

    public static int AddCustomFrameCallback(FrameCallback callback){ return 1; }
    public static int AddCustomFrameSentCallback(FrameSentCallback callback){ return 1; }
    public static void SetUnityDebug(UnityDebugCallback callback){ }
    public static void SetNetworkDelayCallback(DelayCallback callback){ }
    public static void SetSystemDelayCallback(DelayCallback callback){ }
#endif

    [MonoPInvokeCallback(typeof(UnityDebugCallback))]
    public static void UnityShowMessage(string message)
    {
        DebugServer.SendDebugMessage(message);
    }

    [MonoPInvokeCallback(typeof(hl2ss.DelayCallback))]
    public static void NetworkDelayCallback(long delay)
    {
        DebugServer.SendDebugMessage("Network delay: " + delay);
    }

    public static void Initialize(bool enableRM, bool enablePV, bool enableMC, bool enableSI, bool enableRC, bool enableSM, bool enableSU, bool enableVI, bool enableMQ, bool enableEET, bool enableEA)
    {
        InitializeStreamsOnUI((enableRM ? 1U : 0U) | (enablePV ? 2U : 0U) | (enableMC ? 4U : 0U) | (enableSI ? 8U : 0U) | (enableRC ? 16U : 0U) | (enableSM ? 32U : 0U) | (enableSU ? 64U : 0U) | (enableVI ? 128U : 0U) | (enableMQ ? 256U : 0U) | (enableEET ? 512U : 0U) | (enableEA ? 1024U : 0));

        SetUnityDebug(UnityShowMessage);
        SetNetworkDelayCallback(NetworkDelayCallback);
    }

    public static void Print(string str)
    {
        DebugMessage(str);
    }

    public static string GetIPAddress()
    {
        byte[] ipaddress = new byte[16 * 2];
        GetLocalIPv4Address(ipaddress, ipaddress.Length);
        return System.Text.Encoding.Unicode.GetString(ipaddress);
    }

    public static bool UpdateCoordinateSystem()
    {
        var scs = Microsoft.MixedReality.OpenXR.PerceptionInterop.GetSceneCoordinateSystem(Pose.identity);
        if (scs == null) { return false; }
        var unk = Marshal.GetIUnknownForObject(scs);
        bool ret = OverrideWorldCoordinateSystem(unk) != 0;
        Marshal.Release(unk);
        return ret;
    }

    public static bool PullMessage(out uint command, out byte[] data)
    {
        uint size   = MQ_SI_Peek();
        bool status = size != ~0U;
        if (status)
        {
            data = new byte[size];
            MQ_SI_Pop(out command, data);
        }
        else
        {
            data = null;
            command = ~0U;
        }
        return status;
    }

    public static void PushResult(uint value)
    {
        MQ_SO_Push(value);
    }

    public static void AcknowledgeMessage(uint command)
    {
        if (command == ~0U) { MQ_Restart(); }
    }
}
