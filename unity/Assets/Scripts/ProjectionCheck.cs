// using System;
// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using UnityEngine.Pool;
// using System.Reflection;
// using Microsoft.MixedReality.Toolkit;
// using Microsoft.MixedReality.Toolkit.SpatialAwareness;
// using Google.Protobuf;
// using MyBox;

// public class ProjectionCheck : DataProcessor
// {
//     Camera mainCamera = null;
//     [SerializeField] LocatableCamera locatableCamera = null;
//     [SerializeField] float sameLabelDistance = 0.30f;
//     [SerializeField] float sameLabelAngle = 30.0f;
//     List<TestYoloObject> yoloObjects = new List<TestYoloObject>();
//     private IMixedRealitySpatialAwarenessMeshObserver observer = null;
//     [SerializeField] RecognitionVisualization toInstantiate;
    
//     void Start()
//     {
//         observer = CoreServices.GetSpatialAwarenessSystemDataProvider<IMixedRealitySpatialAwarenessMeshObserver>();
//         if (observer == null)
//         {
//             DebugServer.SendDebugMessage("Warning: No mesh observer found");
//         }
//         else
//         {
//             DebugServer.SendDebugMessage("observer: " + observer.Name);
//         }
        
//         mainCamera = Camera.main;
//     }

//     public override void ProcessData(RecognitionData data, uint width, uint height)
//     {
//         // if locatable camera is not set, use the locatable camera in the scene
//         if (locatableCamera == null)
//         {
//             locatableCamera = FindObjectOfType<LocatableCamera>();
//         }
//         // if still not set, throw an error
//         if (locatableCamera == null)
//         {
//             throw new System.Exception("LocatableCamera is not set and cannot be found in the scene");
//         }

//         MainThreadDispatcher.Instance.Enqueue(() =>
//         {
//             ParseRecognition(data, width, height);
//         });
//     }

//     void ParseRecognition(
//         RecognitionData data,
//         uint width,
//         uint height
//     ){
//         InitTags();

//         int numObjects = data.ClassNames.Count;
//         var augmentations = data.Augmentations;

//         var (cameraPosition, cameraRotation) = locatableCamera.GetPosRot(data.Timestamp);

//         for (int i=0; i<numObjects; i++)
//         {
//             // get the class name
//             var class_name = data.ClassNames[i];
//             // get the geometry center
//             var center = new Vector2(data.GeometryCenters[i].X, data.GeometryCenters[i].Y);

//             // get the color
//             Color color = Color.yellow;
//             if (i < data.Colors.Count)
//             {
//                 var dataColor = data.Colors[i];
//                 color = new Color(dataColor.R / 255.0f, dataColor.G / 255.0f, dataColor.B / 255.0f, 1.0f);
//             }

//             Vector3 objectPosition = Vector3.zero, objectPosition3D = Vector3.zero;
//             float hitDistance = 0.0f, hitDistance3D = 0.0f;

//             Vector3 worldSpaceOrigin, worldSpaceDir, worldCameraPosition;

//             // // check if the recognition data contains camera position and rotation
//             // if (data.PoseValid)
//             // {
//             //     var position = data.CameraPosition;
//             //     var rotation = data.CameraRotation;
//             //     var cameraPosition = new Vector3(position.X, position.Y, position.Z);
//             //     var cameraRotation = new Vector3(rotation.X, rotation.Y, rotation.Z);
//             //     worldSpaceOrigin = cameraPosition;
//             //     worldSpaceDir = locatableCamera.PixelCoordToWorldCoord(center, cameraPosition, cameraRotation);
//             // }
//             // else
//             // {
//             //     // from geometry center, get the world space direction of the ray from the camera to the pixel
//             //     worldSpaceOrigin = locatableCamera.transform.position;
//             //     worldSpaceDir = locatableCamera.PixelCoordToWorldCoord(center);
//             // }

//             worldSpaceOrigin = cameraPosition;
//             (worldSpaceDir, worldCameraPosition) = locatableCamera.PixelCoordToWorldCoord(center, cameraPosition, cameraRotation);
//             var (worldSpace2D, worldCameraPosition2D) = locatableCamera.PixelCoordToWorldCoord2D(
//                 center,
//                 cameraPosition,
//                 cameraRotation
//             );

//             // cast a ray from the camera to the pixel
//             RaycastHit hit;
//             TestYoloObject RealtimeYoloObject;
            
//             if (Physics.Raycast(worldSpaceOrigin, worldSpaceDir, out hit, Mathf.Infinity, observer.MeshPhysicsLayerMask)) {
//                 // if the ray hits a surface, use the hit point as the position
//                 objectPosition3D = hit.point;
//                 hitDistance3D = hit.distance;
//             }
//             // else {
//             //     continue;
//             // }
//             objectPosition = worldSpace2D;
//             hitDistance = Vector3.Distance(worldSpace2D, worldCameraPosition2D);
//             RealtimeYoloObject = new TestYoloObject(class_name, objectPosition);

//             // check if the object has been seen before
//             if (SeenBefore(RealtimeYoloObject, worldSpaceOrigin, out int index))
//             {
//                 yoloObjects[index].position = objectPosition;
//                 yoloObjects[index].visualization.ProcessRecognitionResult2D(objectPosition, hitDistance, class_name, center, width, height, color, data, i);
//                 yoloObjects[index].UpdatePosition(objectPosition3D, worldSpace2D);
//             }
//             // else, if there is a collision point
//             else
//             {
//                 // add the yolo object to the list
//                 RecognitionVisualization visObj = Instantiate(toInstantiate, transform);
//                 visObj.ProcessRecognitionResult2D(
//                     objectPosition,
//                     hitDistance,
//                     class_name,
//                     center,
//                     width,
//                     height,
//                     color,
//                     data,
//                     i
//                 );
//                 // visObj.TestProcessRecognitionResult(objectPosition, class_name, score, mask_contour, box, width, height, color);
//                 RealtimeYoloObject.visualization = visObj;
//                 RealtimeYoloObject.Process(objectPosition3D, worldSpace2D);
//                 yoloObjects.Add(RealtimeYoloObject);
//             }
//         }

//         // clear out the tags that are not seen and hanging around for too long
//         ClearTags();
//     }

//     // initialize all tags to be not seen and possibly disappear
//     void InitTags()
//     {
//         for (int i = 0; i < yoloObjects.Count; i++)
//         {
//             yoloObjects[i].lastSeen = false;
//         }

//         // update the visibility of the objects
//         for (int i = 0; i < yoloObjects.Count; i++)
//         {
//             bool lastCanBeSeen = yoloObjects[i].canBeSeen;
//             yoloObjects[i].canBeSeen = CanBeSeen(yoloObjects[i].position);
//             // if the object can be seen now, and it was not seen before
//             if (yoloObjects[i].canBeSeen && !lastCanBeSeen)
//             {
//                 yoloObjects[i].lastSeen = true;
//             }
//         }
//     }

//     void ClearTags(bool clearAll = false)
//     {
//         for (int i = 0; i < yoloObjects.Count; i++)
//         {
//             // if (clearAll || (yoloObjects[i].canBeSeen && !yoloObjects[i].lastSeen))
//             if (clearAll || !yoloObjects[i].lastSeen)
//             {
//                 Destroy(yoloObjects[i].visualization.gameObject);
//                 Destroy(yoloObjects[i].tagBalls);
//                 yoloObjects.RemoveAt(i);
//                 i--;
//             }
//         }
//     }

//     bool SeenBefore(TestYoloObject RealtimeYoloObject, Vector3 cameraPosition, out int index)
//     {
//         bool seenBefore = false;
//         index = -1;
//         for (int i = 0; i < yoloObjects.Count; i++)
//         {
//             // not the same class
//             if (yoloObjects[i].className != RealtimeYoloObject.className)
//             {
//                 continue;
//             }
//             // already seen
//             if (yoloObjects[i].lastSeen)
//             {
//                 continue;
//             }

//             // close enough
//             if (CloseEnough(RealtimeYoloObject, yoloObjects[i], cameraPosition, sameLabelDistance, sameLabelAngle))
//             {
//                 // update last seen time
//                 yoloObjects[i].lastSeen = true;
//                 // set the reference to the visualization object
//                 RealtimeYoloObject.visualization = yoloObjects[i].visualization;
//                 RealtimeYoloObject.tagBalls = yoloObjects[i].tagBalls;
//                 seenBefore = true;
//                 index = i;
//                 break;
//             }
//         }

//         return seenBefore;
//     }

//     bool CloseEnough(TestYoloObject newObject, TestYoloObject oldObject, Vector3 cameraPosition, float distanceThreshold = 0.1f, float angleThreshold = 10.0f)
//     {
//         // first case: the distance is close enough
//         if (Vector3.Distance(newObject.position, oldObject.position) <= distanceThreshold)
//         {
//             return true;
//         }

//         var newDirection = newObject.position - cameraPosition;
//         var oldDirection = oldObject.position - cameraPosition;

//         // second case: regardless of collision, the angle is close enough
//         if (Vector3.Angle(newDirection, oldDirection) <= angleThreshold / 2.0f)
//         {
//             return true;
//         }

//         return false;
//     }

//     void PrintFields(object obj)
//     {
//         PropertyInfo[] properties = obj.GetType().GetProperties();
//         foreach (PropertyInfo property in properties)
//         {
//             Debug.Log($"Field: {property.Name}, Type: {property.PropertyType}, Value: {property.GetValue(obj, null)}");
//         }
//     }

//     bool CanBeSeen(Vector3 position)
//     {
//         Vector3 viewPos = mainCamera.WorldToViewportPoint(position);
//         return viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1 && viewPos.z > 0;
//     }
// }

// class TestYoloObject
// {
//     public string className;
//     public Vector3 position;
//     public bool lastSeen;
//     public bool canBeSeen;
//     public RecognitionVisualization visualization;
//     public GameObject tagBalls;
//     public TestYoloObject(string className, Vector3 position)
//     {
//         this.className = className;
//         this.position = position;
//         this.lastSeen = true;
//         this.canBeSeen = true;
//     }

//     public void Process(Vector3 position3D, Vector3 position2D)
//     {
//         // put a green ball at the 3D position
//         // put a yellow ball at the 2D position
//         tagBalls = GameObject.CreatePrimitive(PrimitiveType.Sphere);
//         tagBalls.transform.position = position3D;
//         tagBalls.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
//         tagBalls.GetComponent<Renderer>().material.color = Color.green;

//         var sphere2D = GameObject.CreatePrimitive(PrimitiveType.Sphere);
//         sphere2D.transform.position = position2D;
//         sphere2D.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
//         sphere2D.GetComponent<Renderer>().material.color = Color.yellow;
//         sphere2D.transform.parent = tagBalls.transform;
//     }

//     public void UpdatePosition(Vector3 position3D, Vector3 position2D)
//     {
//         tagBalls.transform.position = position3D;
//         tagBalls.transform.GetChild(0).transform.position = position2D;
//     }
// }