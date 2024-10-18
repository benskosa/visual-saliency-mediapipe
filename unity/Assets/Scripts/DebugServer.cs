using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class DebugServer : MonoBehaviour
{
    public string address = null;
    public int port = 0;
    string _address = null;
    List<string> _messages = new List<string>();

    static DebugServer _instance = null;
    public static DebugServer Instance
    {
        get
        {
            return _instance;
        }
    }

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
        if (address != null && port != 0)
            _address = address + ":" + port.ToString();
        else
            _address = null;
    }

    void Update()
    {
        // update() is in the main thread
        if (_address == null)
            return;

        while (_messages.Count > 0)
        {
            string msg = _messages[0];
            _messages.RemoveAt(0);
            StartCoroutine(SendDebugCoroutine(msg));
        }
    }
    public void SendDebug(string msg)
    {
        // add time
        msg = System.DateTime.Now.ToString("yyyy/MM/dd HH:mm:ss.fff") + " " + msg;
        _messages.Add(msg);
    }

    IEnumerator SendDebugCoroutine(string msg)
    {
        WWWForm form = new WWWForm();
        form.AddField("message", msg);
        // timeout: 5s
        using (UnityWebRequest www = UnityWebRequest.Post(_address, form))
        {
            www.timeout = 5;
            yield return www.SendWebRequest();
        
            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                if (www.result == UnityWebRequest.Result.ConnectionError)
                {
                    Debug.LogError("Connection error: " + www.error); 
                }
                else if (www.result == UnityWebRequest.Result.ProtocolError)
                {
                    Debug.LogError("Protocol error: " + www.error);
                }
            }
        }
    }

    void _SetAddress(string address, int port = -1)
    {
        if (address[0] >= '0' && address[0] <= '9') {
            address = "http://" + address;
        }
        this.address = address;
        if (port > 0)
            this.port = port;
        _address = this.address + ":" + this.port.ToString();
    }

    public static void SendDebugMessage(string msg)
    {
        Instance.SendDebug(msg);
    }

    public static void SetAddress(string address, int port = -1)
    {
        Instance._SetAddress(address, port);
    }
}
