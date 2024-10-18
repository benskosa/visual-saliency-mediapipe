using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows.WebCam;
using Microsoft.MixedReality.Toolkit;
// using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System.Runtime.InteropServices;
using AOT;
using System;

public class LocatableCamera : MonoBehaviour
{
    private PhotoCapture photoCaptureObject = null;
    private uint cameraResolutionWidth = 640;
    private uint cameraResolutionHeight = 360;
    private uint cameraFramerate = 30;
    private Matrix4x4 projectionMatrix = Matrix4x4.identity;
    private Matrix4x4 inverseProjectionMatrix = Matrix4x4.identity;

    // Start() creates a PhotoCapture object and wait for it to be initialized
    // However, it is possible that Init() is called before Start() is finished
    // In this case, we will save the Init() part in this variable
    // and call it when Start() is finished
    delegate void InitCallback(uint width, uint height, uint fps);
    InitCallback initCallback = null;
    // temporary variable to store the camera's position and rotation provided externally
    [Tooltip("Auxiliary game object to store external camera position and rotation")]
    public GameObject auxiliaryCamera = null;
    GameObject dummyMainCamera = null;
    GameObject dummyPlane = null;
    GameObject dummyObject = null;
    
    // pre-defined local position and rotation of the camera
    Vector3 initialPosition = new Vector3(0.00f, 0.03f, 0.06f);
    Quaternion initialRotation = Quaternion.Euler(4.58f, 359.93f, 0.00f);
    Vector3 externalPosOffset = new Vector3(0.01f, 0.00f, -0.01f);
    Vector3 externalRotOffset = new Vector3(-1.02f, 0.58f, -3.0f);
    float anchor2DDistance = 0f;
    public float anchorPlaneWidth { get; private set; } = 0f;
    public float anchorPlaneHeight { get; private set; } = 0f;
    static LocatableCamera _instance = null;
    // keep a window of the position and rotation of the last 5 seconds
    // estimate: 30fps * 5sec = 150 frames, array length <= 150
    // ulong: milliseconds since the app started
    const int windowSize = 150;
    (ulong, Vector3, Quaternion)[] posRotWindow = new (ulong, Vector3, Quaternion)[windowSize];
    int head = 0;
    int tail = 0;
    float[] projection = new float[3];
    Vector2[] quadCorners = new Vector2[4];
    ulong nextTimestamp;
    bool needUpdate = false;

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

    public void SetProjection(float[] proj)
    {
        if (proj.Length != 3)
        {
            DebugServer.SendDebugMessage("Locatable Camera: Invalid projection matrix");
            return;
        }

        projection = new float[3];
        Array.Copy(proj, projection, 3);
    }

    void FrameArrived(ulong timestamp, float[] pose)
    {
        // remove the old positions and rotations
        while (head != tail && posRotWindow[head].Item1 < timestamp - 3000)
        {
            head = (head + 1) % windowSize;
        }
        // capacity
        if ((tail + 1) % windowSize == head)
        {
            head = (head + 1) % windowSize;
        }

        var (position, rotation) = ComputePosRot(pose);
        posRotWindow[tail] = (timestamp, position, rotation);
        tail = (tail + 1) % windowSize;

        // add the timestamp to the window
        // since it's not in the main thread, we register the timestamp and the pose
        // nextTimestamp = timestamp;
        // needUpdate = true;
    }

    (Vector3, Quaternion) ComputePosRot(float[] pose)
    {
        Vector3 position = new Vector3(pose[12], pose[13], -pose[14]);

        Vector3 row1 = new Vector3(pose[0], pose[1], pose[2]);
        Vector3 row2 = new Vector3(pose[4], pose[5], pose[6]);
        Vector3 row3 = new Vector3(pose[8], pose[9], pose[10]);

        Vector3 pixelDirection = new Vector3(
            projection[0] * row1.x + projection[1] * row2.x + projection[2] * row3.x,
            projection[0] * row1.y + projection[1] * row2.y + projection[2] * row3.y,
            projection[0] * row1.z + projection[1] * row2.z + projection[2] * row3.z
        );

        pixelDirection = Vector3.Normalize(pixelDirection);
        Vector3 rot = new Vector3(-pixelDirection.x, -pixelDirection.y, pixelDirection.z);
        Quaternion rotation = Quaternion.LookRotation(rot);

        return (position, rotation);
    }

    // look up the position and rotation of the camera at a certain time
    public (Vector3, Quaternion) GetPosRot(ulong timestamp)
    {
        // if the window is empty, return the current position and rotation
        if (head == tail)
        {
            return (transform.position, transform.rotation);
        }
        // if the timestamp is older than the oldest position and rotation in the window (unlikely)
        if (timestamp < posRotWindow[head].Item1)
        {
            return (posRotWindow[0].Item2, posRotWindow[0].Item3);
        }
        // if the timestamp is newer than the newest position and rotation in the window
        var lastIdx = (tail - 1 + windowSize) % windowSize;
        if (timestamp > posRotWindow[lastIdx].Item1)
        {
            // if it's too new (>1 sec), return the current position and rotation
            if (timestamp - posRotWindow[lastIdx].Item1 > 1000)
            {
                return (transform.position, transform.rotation);
            }
            return (posRotWindow[lastIdx].Item2, posRotWindow[lastIdx].Item3);
        }
        // otherwise, find the closest position and rotation in the window
        
        // int i = 0;
        // while (i < posRotWindow.Count && posRotWindow[i].Item1 < timestamp)
        // {
        //     i++;
        // }
        // if (i == posRotWindow.Count)
        // {
        //     return (posRotWindow[i - 1].Item2, posRotWindow[i - 1].Item3);
        // }
        // // linear interpolation
        // var (t1, p1, r1) = posRotWindow[i - 1];
        // var (t2, p2, r2) = posRotWindow[i];
        // float t = (float)(timestamp - t1) / (t2 - t1);
        // return (Vector3.Lerp(p1, p2, t), Quaternion.Lerp(r1, r2, t));

        // otherwise, find the closest position and rotation in the window
        int i = head;
        while (i != tail && posRotWindow[i].Item1 < timestamp)
        {
            i = (i + 1) % windowSize;
        }
        if (i == head)
        {
            return (posRotWindow[head].Item2, posRotWindow[head].Item3);
        }
        // simply return the previous position and rotation
        lastIdx = (i - 1 + windowSize) % windowSize;
        return (posRotWindow[lastIdx].Item2, posRotWindow[lastIdx].Item3);
    }

    [MonoPInvokeCallback(typeof(hl2ss.FrameCallback))]
    public static void UpdateCameraReceiveCallback(ulong timestamp, float[] pose)
    {
        if (_instance != null) {
            try {
                _instance.FrameArrived(timestamp, pose);
            }
            catch (Exception e) {
                DebugServer.SendDebugMessage("Locatable Camera: " + e.Message);
            }
        }
    }

    void Start()
    {
        // we want to make this object a child of the camera
        // first create a PhotoCapture object, but don't initialize it yet
        // when called Init(width, height, fps), it will be initialized with the camera's current settings
        // then we can take down and update this camera's position and rotation
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject) {
            photoCaptureObject = captureObject;

            // if Init() was called before Start() is finished, call it now
            if (initCallback != null)
            {
                initCallback(cameraResolutionWidth, cameraResolutionHeight, cameraFramerate);
                initCallback = null;
            }
        });

        // set the position and rotation of this object
        transform.localPosition = initialPosition;
        transform.localRotation = initialRotation;

        // set the projection matrix to a pre-defined value:
        // 1.51550 0.00000 0.01660 0.00000
        // 0.00000 2.69317 -0.03920        0.00000
        // 0.00000 0.00000 -1.00401        -0.20040
        // 0.00000 0.00000 -1.00000        0.00000
        // projectionMatrix = new Matrix4x4(
        //     new Vector4(1.51550f, 0.00000f, 0.01660f, 0.00000f),
        //     new Vector4(0.00000f, 2.69317f, -0.03920f, 0.00000f),
        //     new Vector4(0.00000f, 0.00000f, -1.00401f, -0.20040f),
        //     new Vector4(0.00000f, 0.00000f, -1.00000f, 0.00000f)
        // );
        // inverseProjectionMatrix = projectionMatrix.inverse;

        // Call Init()
        Init(cameraResolutionWidth, cameraResolutionHeight, cameraFramerate);

        if (auxiliaryCamera == null)
        {
            DebugServer.SendDebugMessage("AuxiliaryCamera not found");
        }
        else
        {
            // dummyMainCamera is a child of the auxiliaryCamera named "DummyMainCamera"
            dummyMainCamera = auxiliaryCamera.transform.Find("DummyMainCamera").gameObject;
            if (dummyMainCamera == null)
            {
                DebugServer.SendDebugMessage("DummyMainCamera not found");
            }

            // dummyPlane is a child of the auxiliaryCamera named "DummyPlane"
            dummyPlane = auxiliaryCamera.transform.Find("DummyPlane").gameObject;
            if (dummyPlane == null)
            {
                DebugServer.SendDebugMessage("DummyPlane not found");
            }
            else
            {
                // dummyObject is a child of dummyPlane named "DummyObject"
                dummyObject = dummyPlane.transform.Find("DummyObject").gameObject;
                if (dummyObject == null)
                {
                    DebugServer.SendDebugMessage("DummyObject not found");
                }
            }
        }

        // register the frame callback
        hl2ss.AddCustomFrameCallback(UpdateCameraReceiveCallback);
    }

    public void Init(uint width, uint height, uint fps)
    {
        // set the camera's resolution and framerate
        // this is the resolution and framerate of the video stream
        // the resolution of the photo is set in TakePhoto()
        // the resolution of the video stream is set in the PhotoCapture.CreateAsync() call in Start()
        cameraResolutionWidth = width;
        cameraResolutionHeight = height;
        cameraFramerate = fps;

        // if Init() is called before Start() is finished, save the parameters in this variable
        // and call it when Start() is finished
        if (photoCaptureObject == null)
        {
            initCallback = Init;
            return;
        }

        CameraParameters c = new CameraParameters();
        c.hologramOpacity = 0.0f;
        c.cameraResolutionWidth = (int)cameraResolutionWidth;
        c.cameraResolutionHeight = (int)cameraResolutionHeight;
        c.frameRate = (int)cameraFramerate;
        c.pixelFormat = CapturePixelFormat.BGRA32;

        photoCaptureObject.StartPhotoModeAsync(c, delegate (PhotoCapture.PhotoCaptureResult result) {
            // Take a picture
            photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
        });
    }


    void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        Matrix4x4 cameraToWorldMatrix;
        photoCaptureFrame.TryGetCameraToWorldMatrix(out cameraToWorldMatrix);
        photoCaptureFrame.TryGetProjectionMatrix(Camera.main.nearClipPlane, Camera.main.farClipPlane, out projectionMatrix);
        inverseProjectionMatrix = projectionMatrix.inverse;

        // if in Unity Editor, then reset the position and rotation
        Vector3 cameraPosition = cameraToWorldMatrix.GetColumn(3);
        Quaternion cameraRotation = Quaternion.LookRotation(-cameraToWorldMatrix.GetColumn(2), cameraToWorldMatrix.GetColumn(1));
        if (Application.isEditor)
        {
            // set the position of this object
            transform.position = cameraPosition;
            // set the rotation of this object
            transform.rotation = cameraRotation;
        }
        // stop the photo mode
        photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);

        // // DEBUG: take another photo
        // photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);

        quadCorners = new Vector2[4] {
            new Vector2(0, 0),
            new Vector2(cameraResolutionWidth, 0),
            new Vector2(cameraResolutionWidth, cameraResolutionHeight),
            new Vector2(0, cameraResolutionHeight)
        };

        Vector3[] worldCorners = new Vector3[4];
        for (int i = 0; i < 4; i++)
        {
            Vector3 worldSpaceDir = PixelCoordToWorldCoord(quadCorners[i]).Item1;
            Vector3 worldSpacePoint = transform.position + worldSpaceDir * 1.0f;
            worldCorners[i] = worldSpacePoint;
        }

        Vector3 videoRendererPosition = Vector3.zero;
        for (int i = 0; i < 4; i++)
        {
            videoRendererPosition += worldCorners[i];
        }
        videoRendererPosition /= 4.0f;

        auxiliaryCamera.transform.position = transform.position;
        auxiliaryCamera.transform.rotation = transform.rotation;
        // dummy main camera reflects the position and rotation of the main camera
        dummyMainCamera.transform.position = Camera.main.transform.position;
        dummyMainCamera.transform.rotation = Camera.main.transform.rotation;

        // in the case of 2D rendering, there is a static offset between the camera and the object
        anchor2DDistance = Vector3.Distance(videoRendererPosition, transform.position);
        anchorPlaneWidth = Vector3.Distance(worldCorners[1], worldCorners[0]);
        anchorPlaneHeight = Vector3.Distance(worldCorners[3], worldCorners[0]);

        // set the position and rotation of the dummy plane
        dummyPlane.transform.position = videoRendererPosition;
        // plane should face the dummy main camera
        Vector3 planeDirection = (dummyPlane.transform.position - dummyMainCamera.transform.position).normalized;
        dummyPlane.transform.rotation = Quaternion.LookRotation(planeDirection, Vector3.up);
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        // Shutdown the photo capture resource
        photoCaptureObject.Dispose();
        photoCaptureObject = null;
    }

    void OnApplicationQuit()
    {
        if (photoCaptureObject != null)
        {
            photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
        }
    }

    public void SetParams(uint width, uint height, uint fps)
    {
        cameraResolutionWidth = width;
        cameraResolutionHeight = height;
        cameraFramerate = fps;
    }

    // internal function: convert a pixel coordinate to a world coordinate
    // providing a Matrix4x4 cameraToWorldMatrix, so that if an external
    // camera position and rotation is provided, we will compute a separate
    // projection matrix based on that, while if not provided, we will use
    // the projection matrix from the camera
    Vector3 _PixelCoordToWorldCoord(Vector2 pixelCoord, Matrix4x4 cameraToWorldMatrix)
    {
        // convert the pixel coordinate to normalized device coordinate (NDC)
        Vector2 ndcCoord = ConvertPixelToNDC(pixelCoord);

        float focalLengthX = projectionMatrix.GetColumn(0).x;
        float focalLengthY = projectionMatrix.GetColumn(1).y;
        float centerX = projectionMatrix.GetColumn(2).x;
        float centerY = projectionMatrix.GetColumn(2).y;

        // normalize factor
        float normFactor = projectionMatrix.GetColumn(2).z;
        centerX /= normFactor;
        centerY /= normFactor;

        // convert the NDC to camera space
        Vector3 dirRay = new Vector3((ndcCoord.x - centerX) / focalLengthX, (ndcCoord.y - centerY) / focalLengthY, -1.0f / normFactor);
        Vector3 direction = new Vector3(
            Vector3.Dot(cameraToWorldMatrix.GetRow(0), dirRay),
            Vector3.Dot(cameraToWorldMatrix.GetRow(1), dirRay),
            Vector3.Dot(cameraToWorldMatrix.GetRow(2), dirRay)
        );

        return direction.normalized;
    }

    // convert a pixel coordinate to a world coordinate
    // the pixel coordinate is relative to the top-left corner of the camera image
    // return: the world space direction of the ray from the camera to the pixel
    // the caller can use this direction to cast a ray from the camera to the pixel
    // camera's position is transform.position of this object
    public (Vector3, Vector3) PixelCoordToWorldCoord(Vector2 pixelCoord)
    {
        Matrix4x4 cameraToWorldMatrix = transform.localToWorldMatrix;
        return (_PixelCoordToWorldCoord(pixelCoord, cameraToWorldMatrix), Camera.main.transform.position);
    }

    void UpdateAuxCamera(Vector3 cameraPosition, Quaternion cameraRotation)
    {
        auxiliaryCamera.transform.position = cameraPosition + externalPosOffset;
        auxiliaryCamera.transform.rotation = cameraRotation;
        var finalRot = auxiliaryCamera.transform.rotation.eulerAngles + externalRotOffset;
        // set the z-axis to be the same as the internal camera
        auxiliaryCamera.transform.rotation = Quaternion.Euler(finalRot.x, finalRot.y, transform.rotation.eulerAngles.z);
    }

    // convert a pixel coordinate to a world coordinate
    // providing a camera position and rotation
    // return: the world space direction of the ray from the camera to the pixel
    public (Vector3, Vector3) PixelCoordToWorldCoord(Vector2 pixelCoord, Vector3 cameraPosition, Vector3 cameraRotation)
    {
        return PixelCoordToWorldCoord(pixelCoord, cameraPosition, Quaternion.LookRotation(cameraRotation));
    }

    public (Vector3, Vector3) PixelCoordToWorldCoord(Vector2 pixelCoord, Vector3 cameraPosition, Quaternion cameraRotation)
    {
        if (auxiliaryCamera == null)
        {
            // Alas! The external camera is not found, downgrading to the internal camera
            DebugServer.SendDebugMessage("AuxiliaryCamera not found, downgrading to the internal camera");
            return PixelCoordToWorldCoord(pixelCoord);
        }
        
        UpdateAuxCamera(cameraPosition, cameraRotation);

        var cameraToWorldMatrix = auxiliaryCamera.transform.localToWorldMatrix;
        return (_PixelCoordToWorldCoord(pixelCoord, cameraToWorldMatrix), auxiliaryCamera.transform.position);
    }

    public (Vector3, Vector3) PixelCoordToWorldCoord2D(Vector2 pixelCoord, Vector3 cameraPosition, Quaternion cameraRotation)
    {
        if (auxiliaryCamera == null)
        {
            // Alas! The external camera is not found, downgrading to the internal camera
            DebugServer.SendDebugMessage("AuxiliaryCamera not found, downgrading to the internal camera");
            return PixelCoordToWorldCoord(pixelCoord);
        }

        UpdateAuxCamera(cameraPosition, cameraRotation);

        // get the magnitude of the x and y offset of the point relative to the center
        // float diff_x = (pixelCoord.x - cameraResolutionWidth / 2.0f) / cameraResolutionWidth * anchorPlaneWidth;
        // float diff_y = (pixelCoord.y - cameraResolutionHeight / 2.0f) / cameraResolutionHeight * anchorPlaneHeight;
        // get the position of the point on the plane
        // dummyObject.transform.localPosition = new Vector3(diff_x, -diff_y, 0.0f); // the y-axis is inverted

        Vector3[] worldCorners = new Vector3[4];
        for (int i = 0; i < 4; i++)
        {
            Vector3 worldSpaceDir = PixelCoordToWorldCoord(quadCorners[i], cameraPosition, cameraRotation).Item1;
            Vector3 worldSpacePoint = transform.position + worldSpaceDir * 1.0f;
            worldCorners[i] = worldSpacePoint;
        }

        Vector3 worldPosition = worldCorners[0] + (worldCorners[1] - worldCorners[0]) * pixelCoord.x / cameraResolutionWidth + (worldCorners[3] - worldCorners[0]) * pixelCoord.y / cameraResolutionHeight;

        // get the world position of the point
        return (worldPosition, dummyMainCamera.transform.position);
    }

    Vector2 ConvertPixelToNDC(Vector2 pixelCoord)
    {
        float halfWidth = cameraResolutionWidth / 2.0f;
        float halfHeight = cameraResolutionHeight / 2.0f;

        pixelCoord.x = (pixelCoord.x - halfWidth) / halfWidth;
        pixelCoord.y = (pixelCoord.y - halfHeight) / halfHeight;

        pixelCoord.y = -pixelCoord.y;

        return pixelCoord;
    }

    void Update()
    {
        if (needUpdate)
        {
            posRotWindow[tail] = (nextTimestamp, transform.position, transform.rotation);
            tail = (tail + 1) % windowSize;
            needUpdate = false;
        }
    }
}
