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

public class RealtimeMarker : DataProcessor
{
    Camera mainCamera = null;
    [SerializeField] LocatableCamera locatableCamera = null;
    [SerializeField] RecognitionVisualization visualizationPrefab = null;
    [SerializeField] float sameLabelDistance = 0.30f;
    [SerializeField] float sameLabelAngle = 30.0f;
    List<RealtimeYoloObject> yoloObjects = new List<RealtimeYoloObject>();
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
    [SerializeField] List<AugmentationTag> augmentationTags = new List<AugmentationTag>();
    Dictionary<string, RecognitionVisualization> augmentationTagDict = new Dictionary<string, RecognitionVisualization>();
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
        foreach (var tag in augmentationTags)
        {
            augmentationTagDict[tag.tag] = tag.visualization;
        }
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

    void ParseRecognition(
        RecognitionData data,
        uint width,
        uint height
    ){
        InitTags();

        int numObjects = data.ClassNames.Count;
        var augmentations = data.Augmentations;

        var (cameraPosition, cameraRotation) = locatableCamera.GetPosRot(data.Timestamp);

        for (int i=0; i<numObjects; i++)
        {
            // get the class name
            var class_name = data.ClassNames[i];
            // get the geometry center
            var center = new Vector2(data.GeometryCenters[i].X, data.GeometryCenters[i].Y);

            // get the color
            Color color = Color.yellow;
            if (i < data.Colors.Count)
            {
                var dataColor = data.Colors[i];
                var alpha = dataColor.A < 5? 1.0f : (float)dataColor.A / 100.0f;
                color = new Color(dataColor.R / 255.0f, dataColor.G / 255.0f, dataColor.B / 255.0f, alpha);
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
            string augmentationTag = "outline"; // hardcoded default
            // if there is a corresponding augmentation tag
            // the augmentation field is optional, so we need to check if it exists
            if (augmentations.Count > i)
            {
                string augmentation = augmentations[i]; // fallback option
                if (augmentationTagDict.ContainsKey(augmentation))
                {
                    toInstantiate = augmentationTagDict[augmentation];
                    augmentationTag = augmentation;
                }
                else
                {
                    DebugServer.SendDebugMessage("Augmentation tag not found: " + augmentation);
                }
            }

            worldSpaceOrigin = cameraPosition;
            (worldSpaceDir, worldCameraPosition) = locatableCamera.PixelCoordToWorldCoord(center, cameraPosition, cameraRotation);

            // cast a ray from the camera to the pixel
            RaycastHit hit;
            RealtimeYoloObject RealtimeYoloObject;
            if ((!render2D) && observer != null && Physics.Raycast(worldSpaceOrigin, worldSpaceDir, out hit, Mathf.Infinity, observer.MeshPhysicsLayerMask))
            {
                // if the ray hits a surface, use the hit point as the position
                objectPosition = hit.point;
                hitDistance = hit.distance;
                RealtimeYoloObject = new RealtimeYoloObject(class_name, objectPosition, augmentationTag);
            }
            else
            {
                if (render2D)
                {
                    objectPosition = worldSpaceOrigin + worldSpaceDir * render2DDistance;
                    hitDistance = render2DDistance;
                    // (objectPosition, worldCameraPosition) = locatableCamera.PixelCoordToWorldCoord2D(center, cameraPosition, cameraRotation);
                    // hitDistance = Vector3.Distance(objectPosition, worldCameraPosition);
                    RealtimeYoloObject = new RealtimeYoloObject(class_name, objectPosition, augmentationTag);
                }
                else if (renderUncollided)
                {
                    objectPosition = worldSpaceOrigin + worldSpaceDir * uncollidedDistance;
                    hitDistance = uncollidedDistance;
                    RealtimeYoloObject = new RealtimeYoloObject(class_name, objectPosition, augmentationTag);
                }
                else
                {
                    continue;
                }
            }

            // check if the object has been seen before
            if (SeenBefore(RealtimeYoloObject, worldSpaceOrigin, out int index))
            {
                yoloObjects[index].position = objectPosition;
                yoloObjects[index].visualization.ProcessRecognitionResult(
                    objectPosition,
                    hitDistance,
                    class_name,
                    center,
                    width,
                    height,
                    color,
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
                    class_name,
                    center,
                    width,
                    height,
                    color,
                    data,
                    i
                );
                // visObj.TestProcessRecognitionResult(objectPosition, class_name, score, mask_contour, box, width, height, color);
                RealtimeYoloObject.visualization = visObj;
                // add the yolo object to the list
                yoloObjects.Add(RealtimeYoloObject);
            }
        }

        // clear out the tags that are not seen and hanging around for too long
        ClearTags();
    }

    // initialize all tags to be not seen and possibly disappear
    void InitTags()
    {
        for (int i = 0; i < yoloObjects.Count; i++)
        {
            yoloObjects[i].lastSeen = false;
        }

        // update the visibility of the objects
        for (int i = 0; i < yoloObjects.Count; i++)
        {
            bool lastCanBeSeen = yoloObjects[i].canBeSeen;
            yoloObjects[i].canBeSeen = CanBeSeen(yoloObjects[i].position);
            // if the object can be seen now, and it was not seen before
            if (yoloObjects[i].canBeSeen && !lastCanBeSeen)
            {
                yoloObjects[i].lastSeen = true;
            }
        }
    }

    void ClearTags(bool clearAll = false)
    {
        for (int i = 0; i < yoloObjects.Count; i++)
        {
            // if (clearAll || (yoloObjects[i].canBeSeen && !yoloObjects[i].lastSeen))
            if (clearAll || !yoloObjects[i].lastSeen)
            {
                Destroy(yoloObjects[i].visualization.gameObject);
                yoloObjects.RemoveAt(i);
                i--;
            }
        }
    }

    bool SeenBefore(RealtimeYoloObject RealtimeYoloObject, Vector3 cameraPosition, out int index)
    {
        bool seenBefore = false;
        index = -1;
        for (int i = 0; i < yoloObjects.Count; i++)
        {
            // not the same class
            if (yoloObjects[i].className != RealtimeYoloObject.className)
            {
                continue;
            }
            // already seen
            if (yoloObjects[i].lastSeen)
            {
                continue;
            }
            // different visualization tag
            if (yoloObjects[i].visualizationTag != RealtimeYoloObject.visualizationTag)
            {
                continue;
            }

            // close enough
            if (CloseEnough(RealtimeYoloObject, yoloObjects[i], cameraPosition, sameLabelDistance, sameLabelAngle))
            {
                // update last seen time
                yoloObjects[i].lastSeen = true;
                // set the reference to the visualization object
                RealtimeYoloObject.visualization = yoloObjects[i].visualization;
                seenBefore = true;
                index = i;
                break;
            }
        }

        return seenBefore;
    }

    bool CloseEnough(RealtimeYoloObject newObject, RealtimeYoloObject oldObject, Vector3 cameraPosition, float distanceThreshold = 0.1f, float angleThreshold = 10.0f)
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

class RealtimeYoloObject
{
    public string className;
    public Vector3 position;
    public bool lastSeen;
    public RecognitionVisualization visualization = null;
    public string visualizationTag = "";
    public bool canBeSeen;
    public Transform transform
    {
        get {return visualization.transform;}
    }
    public RealtimeYoloObject(string className, Vector3 position, string visualizationTag = "")
    {
        this.className = className;
        this.position = position;
        this.lastSeen = true;
        this.canBeSeen = true;
        this.visualizationTag = visualizationTag;
    }
}