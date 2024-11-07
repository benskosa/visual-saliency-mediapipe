using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Pool;
using System.Reflection;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using Google.Protobuf;
using MyBox;

using Assets.Scripts.DataTypes.FaceMesh;  // For FaceMeshColors

public class RealtimeMarker : DataProcessor
{
    Camera mainCamera = null;
    [SerializeField] LocatableCamera locatableCamera = null;
    [SerializeField] RecognitionVisualization visualizationPrefab = null;
    [SerializeField] float sameLabelDistance = 0.30f;
    [SerializeField] float sameLabelAngle = 30.0f;
    // List<RealtimeYoloObject> yoloObjects = new List<RealtimeYoloObject>();
    List<RealtimeFaceMaskObject> faceMeshObjects = new List<RealtimeFaceMaskObject>();

    private IMixedRealitySpatialAwarenessMeshObserver observer = null;
    [SerializeField] bool renderUncollided = false;
    // if true, specify the distance
    [ConditionalField(nameof(renderUncollided))] [SerializeField] float uncollidedDistance = 5.0f;
    [SerializeField] bool render2D = false;
    [ConditionalField(nameof(render2D))] [SerializeField] float render2DDistance = 5.0f;
    // map from string to RecognitionVisualization
    // use a workaround to expose it in the inspector
    [Serializable] public struct AugmentationTag
    {
        public string tag;
        public RecognitionVisualization visualization;
    }
    // AugmentationTags
    // [SerializeField] List<AugmentationTag> augmentationTags = new List<AugmentationTag>();  // Ben Removed
    // Dictionary<string, RecognitionVisualization> augmentationTagDict = new Dictionary<string, RecognitionVisualization>();  // Ben Removed
    ulong lastTimestamp = 0;

    void Start()
    {
        // get the mesh observer profile
        if (!render2D) {
            observer = CoreServices.GetSpatialAwarenessSystemDataProvider<IMixedRealitySpatialAwarenessMeshObserver>();
            if (observer == null)
            {
                DebugServer.SendDebugMessage("Warning: No mesh observer found");
            }
            else
            {
                DebugServer.SendDebugMessage("observer: " + observer.Name);
            }
        }

        mainCamera = Camera.main;

        // convert the list to a dictionary
        // AugmentationTags
        // foreach (var tag in augmentationTags)
        // {
        //     augmentationTagDict[tag.tag] = tag.visualization;
        // }
    }

    public override void ProcessData(RecognitionData data, uint width, uint height)
    {
        // if locatable camera is not set, use the locatable camera in the scene
        if (locatableCamera == null)
        {
            locatableCamera = FindObjectOfType<LocatableCamera>();
        }
        // if still not set, throw an error
        if (locatableCamera == null)
        {
            throw new System.Exception("LocatableCamera is not set and cannot be found in the scene");
        }

        // if the timestamp is not newer than last one, skip
        if (data.Timestamp <= lastTimestamp)
        {
            return;
        }

        MainThreadDispatcher.Instance.Enqueue(() =>
        {
            ParseRecognition(data, width, height);
        });
    }


    /// <summary>
    /// Iterates through each object detected in the data we recieved and processes
    /// each:
    ///     1. if we want to get z value for the object (i.e. !render2d), then shoot a ray
    ///     from the HoloLens to the mesh that's already been laid by the depth sensor
    ///     and have the z value be where the ray and meshes collide
    ///     2. Either update the z-value of the GameObject for the recognized object
    ///     (of type RecognitionVisualization) or create a new GameObject for the recognized
    ///     object. After it is created, it should draw itself.
    /// </summary>
    /// <param name="data">The data we recieved from the recognition model.</param>
    /// <param name="width">width of the current frame.</param>
    /// <param name="height">height of the current frame.</param>
    void ParseRecognition(
        RecognitionData data,
        uint width,
        uint height
    ){
        // Could "cheat" and use prior knowledge about "at 1 meter away, the average human face
        // is x wide, y tall...and since the face points are n wide and m tall, then we know that
        // the face is approx i meters away."
        InitTags();

        int numFaces = data.Faces.Count;
        int numFaceAugmentations = data.FaceMeshColors[0].Count;  // Should be 8

        var (cameraPosition, cameraRotation) = locatableCamera.GetPosRot(data.Timestamp);

        for (int i=0; i<numFaces; i++) {
            // For every face, we want to find the approx distance from the center
            // get the geometry center. We are treating a point on the nose as a Geometry center for now,
            // specifically index 4 as shown on https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
            // var center = new Vector2(data.GeometryCenters[i].X, data.GeometryCenters[i].Y);
            var center = new Vector2(data.Faces[i][4].X, data.Faces[i][4].Y);

            // for this face, go through every type of landmark and get it's color
            FaceMeshColors faceMeshColors = FaceMeshColors();
            if (i < data.FaceMeshColors.Count)
            {
                for (int j = 0; j < numFaceAugmentations; j++) {
                    var faceMeshAugColors = data.FaceMeshColors[j];
                    var alpha = faceMeshAugColors.a < 5? 1.0f : (float)faceMeshAugColors.a / 100.0f;
                    faceMeshColors.FaceMeshTesselationColor = new Color(faceMeshAugColors.r / 255.0f, faceMeshAugColors.g / 255.0f, faceMeshAugColors.b / 255.0f, alpha);
                }
            }

            Vector3 objectPosition = Vector3.zero;
            float hitDistance = 0.0f;

            Vector3 worldSpaceOrigin, worldSpaceDir, worldCameraPosition;

            // // check if the recognition data contains camera position and rotation
            // if (data.PoseValid)
            // {
            //     var position = data.CameraPosition;
            //     var rotation = data.CameraRotation;
            //     var cameraPosition = new Vector3(position.X, position.Y, position.Z);
            //     var cameraRotation = new Vector3(rotation.X, rotation.Y, rotation.Z);
            //     worldSpaceOrigin = cameraPosition;
            //     worldSpaceDir = locatableCamera.PixelCoordToWorldCoord(center, cameraPosition, cameraRotation);
            // }
            // else
            // {
            //     // from geometry center, get the world space direction of the ray from the camera to the pixel
            //     worldSpaceOrigin = locatableCamera.transform.position;
            //     worldSpaceDir = locatableCamera.PixelCoordToWorldCoord(center);
            // }

            RecognitionVisualization toInstantiate = visualizationPrefab;
            // AugmentationTags
            // string augmentationTag = "outline"; // hardcoded default
            // if there is a corresponding augmentation tag
            // the augmentation field is optional, so we need to check if it exists
            // if (augmentations.Count > i) {
            //     string augmentation = augmentations[i]; // fallback option
            //     if (augmentationTagDict.ContainsKey(augmentation))
            //     {
            //         toInstantiate = augmentationTagDict[augmentation];
            //         augmentationTag = augmentation;
            //     }
            //     else
            //     {
            //         DebugServer.SendDebugMessage("Augmentation tag not found: " + augmentation);
            //     }
            // }

            worldSpaceOrigin = cameraPosition;
            (worldSpaceDir, worldCameraPosition) = locatableCamera.PixelCoordToWorldCoord(center, cameraPosition, cameraRotation);

            // cast a ray from the camera to the pixel
            RaycastHit hit;
            RealtimeFaceMaskObject realtimeFaceMaskObject;
            if ((!render2D) && observer != null && Physics.Raycast(worldSpaceOrigin, worldSpaceDir, out hit, Mathf.Infinity, observer.MeshPhysicsLayerMask))
            {
                // if the ray hits a surface, use the hit point as the position
                objectPosition = hit.point;
                hitDistance = hit.distance;
                realtimeFaceMaskObject = new RealtimeFaceMaskObject(objectPosition);
                // realtimeFaceMaskObject = new RealtimeYoloObject(class_name, objectPosition, augmentationTag);

            } else {
                if (render2D)  // If we don't want to use depth/raycast, then just use hardcoded depth (base z value)
                {
                    objectPosition = worldSpaceOrigin + worldSpaceDir * render2DDistance;
                    hitDistance = render2DDistance;
                    // (objectPosition, worldCameraPosition) = locatableCamera.PixelCoordToWorldCoord2D(center, cameraPosition, cameraRotation);
                    // hitDistance = Vector3.Distance(objectPosition, worldCameraPosition);
                    // realtimeFaceMaskObject = new RealtimeYoloObject(class_name, objectPosition, augmentationTag);
                    realtimeFaceMaskObject = new RealtimeFaceMaskObject(objectPosition);
                }
                else if (renderUncollided)
                {
                    objectPosition = worldSpaceOrigin + worldSpaceDir * uncollidedDistance;
                    hitDistance = uncollidedDistance;
                    // realtimeFaceMaskObject = new RealtimeYoloObject(class_name, objectPosition, augmentationTag);
                    realtimeFaceMaskObject = new RealtimeFaceMaskObject(objectPosition);

                }
                else
                {
                    continue;
                }
            }

            // check if the object has been seen before
            if (SeenBefore(realtimeFaceMaskObject, worldSpaceOrigin, out int index))
            {
                faceMeshObjects[index].position = objectPosition;
                faceMeshObjects[index].visualization.ProcessRecognitionResult(
                    objectPosition,
                    hitDistance,
                    // class_name,
                    center,
                    width,
                    height,
                    faceMeshColors,
                    data,
                    i
                );
            }
            // else, if there is a collision point
            else
            {
                // instantiate a new yolo object label
                RecognitionVisualization visObj = Instantiate(toInstantiate, transform);
                visObj.ProcessRecognitionResult(
                    objectPosition,
                    hitDistance,
                    // class_name,
                    center,
                    width,
                    height,
                    faceMeshColors,
                    data,
                    i
                );
                // visObj.TestProcessRecognitionResult(objectPosition, class_name, score, mask_contour, box, width, height, color);
                global::RealtimeFaceMaskObject.visualization = visObj;
                // add the face mesh object to the list
                faceMeshObjects.Add(realtimeFaceMaskObject);
            }
        }

        // clear out the tags that are not seen and hanging around for too long
        ClearTags();
    }

    // initialize all tags to be not seen and possibly disappear
    void InitTags()
    {
        for (int i = 0; i < faceMeshObjects.Count; i++)
        {
            faceMeshObjects[i].lastSeen = false;
        }

        // update the visibility of the objects
        for (int i = 0; i < faceMeshObjects.Count; i++)
        {
            bool lastCanBeSeen = faceMeshObjects[i].canBeSeen;
            faceMeshObjects[i].canBeSeen = CanBeSeen(faceMeshObjects[i].position);
            // if the object can be seen now, and it was not seen before
            if (faceMeshObjects[i].canBeSeen && !lastCanBeSeen)
            {
                faceMeshObjects[i].lastSeen = true;
            }
        }
    }

    void ClearTags(bool clearAll = false)
    {
        for (int i = 0; i < faceMeshObjects.Count; i++)
        {
            // if (clearAll || (yoloObjects[i].canBeSeen && !yoloObjects[i].lastSeen))
            if (clearAll || !faceMeshObjects[i].lastSeen)
            {
                Destroy(faceMeshObjects[i].visualization.gameObject);
                faceMeshObjects.RemoveAt(i);
                i--;
            }
        }
    }

    bool SeenBefore(RealtimeFaceMaskObject RealtimeFaceMaskObject, Vector3 cameraPosition, out int index)
    {
        bool seenBefore = false;
        index = -1;
        for (int i = 0; i < faceMeshObjects.Count; i++)
        {
            // Ben: class names and visualization tags don't apply here
            // not the same class
            // if (faceMeshObjects[i].className != RealtimeFaceMaskObject.className)
            // {
            //     continue;
            // }
            // already seen
            if (faceMeshObjects[i].lastSeen)
            {
                continue;
            }
            // // different visualization tag
            // if (faceMeshObjects[i].visualizationTag != RealtimeFaceMaskObject.visualizationTag)
            // {
            //     continue;
            // }

            // close enough
            if (CloseEnough(RealtimeFaceMaskObject, faceMeshObjects[i], cameraPosition, sameLabelDistance, sameLabelAngle))
            {
                // update last seen time
                faceMeshObjects[i].lastSeen = true;
                // set the reference to the visualization object
                RealtimeFaceMaskObject.visualization = faceMeshObjects[i].visualization;
                seenBefore = true;
                index = i;
                break;
            }
        }

        return seenBefore;
    }

    bool CloseEnough(RealtimeFaceMaskObject newObject, RealtimeFaceMaskObject oldObject, Vector3 cameraPosition, float distanceThreshold = 0.1f, float angleThreshold = 10.0f)
    {
        // first case: the distance is close enough
        if (Vector3.Distance(newObject.position, oldObject.position) <= distanceThreshold)
        {
            return true;
        }

        var newDirection = newObject.position - cameraPosition;
        var oldDirection = oldObject.position - cameraPosition;

        // second case: regardless of collision, the angle is close enough
        if (Vector3.Angle(newDirection, oldDirection) <= angleThreshold / 2.0f)
        {
            return true;
        }

        return false;
    }

    void PrintFields(object obj)
    {
        PropertyInfo[] properties = obj.GetType().GetProperties();
        foreach (PropertyInfo property in properties)
        {
            Debug.Log($"Field: {property.Name}, Type: {property.PropertyType}, Value: {property.GetValue(obj, null)}");
        }
    }

    bool CanBeSeen(Vector3 position)
    {
        Vector3 viewPos = mainCamera.WorldToViewportPoint(position);
        return viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1 && viewPos.z > 0;
    }
}

// TODO for Ben: Figure out if I can use this for face landmark visualization
class RealtimeFaceMaskObject
{
    // public string className;
    public Vector3 position;
    public bool lastSeen;
    // public RecognitionVisualization visualization = null;
    public MediapipeVisualization visualization = null;
    // public string visualizationTag = "";
    public bool canBeSeen;
    public Transform transform
    {
        get {return visualization.transform;}
    }
    // public RealtimeFaceMaskObject(string className, Vector3 position, string visualizationTag = "")
    public RealtimeFaceMaskObject(Vector3 position)
    {
        // this.className = className;
        this.position = position;
        this.lastSeen = true;
        this.canBeSeen = true;
        // this.visualizationTag = visualizationTag;
    }
}

// class RealtimeYoloObject
// {
//     public string className;
//     public Vector3 position;
//     public bool lastSeen;
//     public RecognitionVisualization visualization = null;
//     public string visualizationTag = "";
//     public bool canBeSeen;
//     public Transform transform
//     {
//         get {return visualization.transform;}
//     }
//     public RealtimeYoloObject(string className, Vector3 position, string visualizationTag = "")
//     {
//         this.className = className;
//         this.position = position;
//         this.lastSeen = true;
//         this.canBeSeen = true;
//         this.visualizationTag = visualizationTag;
//     }
// }