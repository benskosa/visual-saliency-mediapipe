using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MyBox;
using System;

using Assets.Scripts.DataTypes.FaceMesh;  // For FaceMeshConnections AND FaceColors

public enum RenderMode
{
    Solid,
    Outline,
    FlashSolid,
    FlashOutline,
};

/// <summary>
///  Class representing a recognized YOLO (or REMDet) object and its augmentation
/// </summary>

public class FaceMaskCanvas : MediapipeVisualization
{
    RectTransform rectTransform;
    RawImage rawImage;
    public RenderMode renderMode = RenderMode.Solid;
    // check if it's Outline or FlashOutline
    // if the render mode is contour, the width of the contour
    [ConditionalField("renderMode", false, RenderMode.Outline, RenderMode.FlashOutline)] public int contourWidth = 3;
    Camera mainCamera = null;
    LocatableCamera locatableCamera = null;
    Texture2D texture = null;
    const int MARGIN = 5; // margin for the contour in case it goes out of the bounding box

    void Awake()
    {
        rectTransform = GetComponent<RectTransform>();
        rawImage = GetComponentInChildren<RawImage>();
        mainCamera = Camera.main;
        // locatable camera is a child of main camera
        locatableCamera = mainCamera.GetComponentInChildren<LocatableCamera>();
    }

    // Renders the recognized object, which in this case is the mesh for the 
    // recognized face.
    //
    // - position: the position of the object, in Unity world space
    // - distance: the distance of the object from the camera
    // - center: the geometry center of the object
    // - width: the width of the image
    // - height: the height of the image
    // - color: the color of the object
    // - data: the recognition data
    //      Sidenote: this isn't the best way to pass data, but no need to parse the contour if not needed
    //      The other passed fields are either computed by caller or used by caller anyways
    // - index: the index of the object in the recognition result
    public override void ProcessRecognitionResult(
        Vector3 position,
        float distance,
        // string class_name,
        Vector2 center,
        uint width,
        uint height,
        FaceColors colors,
        RecognitionData data,
        int index
    )
    {
        // set position
        transform.position = position;

        // be perpendicular to the locatable camera
        var forward = locatableCamera? locatableCamera.transform.forward : mainCamera.transform.forward;
        transform.forward = forward;
        // transform.Rotate(0, 180, 0);

        // overlay the z rotation of the camera
        var zRotation = locatableCamera? locatableCamera.transform.rotation.eulerAngles.z : mainCamera.transform.rotation.eulerAngles.z;
        transform.RotateAround(transform.position, transform.forward, zRotation);

        // parse the other data needed
        var box = data.Boxes[index];

        // bounding box
        var bounding_box = new int[4];
        bounding_box[0] = box.X1;
        bounding_box[1] = box.Y1;
        bounding_box[2] = box.X2;
        bounding_box[3] = box.Y2;

        // render the recognition result
        RenderRecognitionResult(position, distance, bounding_box, center, width, height, colors, data, index);
    }

    // Renders the recognized object, which in this case is the mesh for the 
    // recognized face.
    // - position: the position of the object, in Unity world space
    // - distance: the distance of the object from the camera
    // - bounding_box: the bounding box corresponding to the recognized face,
    //                 which is in the format [x1, y1, x2, y2].
    // - center: the geometry center of the object
    // - width: the width of the image
    // - height: the height of the image
    // - color: the color of the object
    // - data: the recognition data
    //      Sidenote: this isn't the best way to pass data, but no need to parse the contour if not needed
    //      The other passed fields are either computed by caller or used by caller anyways
    // - index: the index of the object in the recognition result
    void RenderRecognitionResult(
        Vector3 position,
        float distance,
        int[] bounding_box,
        Vector2 center,
        uint width,
        uint height,
        FaceColors colors,
        RecognitionData data,
        int index
    )
    {
        // Ben's Game Plan (Nov 5, 2024):
        //  - Create a function that just draws edges that you define (pass in a list of edges)
        //    in the color, thickness, etc that you pass in.
        //  - Use the edge mappings for the different parts of the face mask that Mediapipe Library
        //    already has and call function for each one of those lists.

        int x1 = bounding_box[0];
        int y1 = bounding_box[1];
        int x2 = bounding_box[2];
        int y2 = bounding_box[3];

        // create a texture for the bounding box
        int boxWidth = x2 - x1;
        int boxHeight = y2 - y1;
        int canvasWidth = boxWidth + MARGIN * 2;
        int canvasHeight = boxHeight + MARGIN * 2;

        if (texture != null)
        {
            Destroy(texture);
        }
        texture = new Texture2D(canvasWidth, canvasHeight, TextureFormat.RGBA32, false);
        int numBytes = canvasWidth * canvasHeight * 4;
        byte[] pixels = new byte[numBytes]; // default value is 0
        // set wrap mode to clamp
        texture.wrapMode = TextureWrapMode.Clamp;

        var face = data.Faces[index].Landmarks;
        // Vector2[] face_landmarks = new Vector2[face.Count];
        Vector3[] face_landmarks = new Vector3[face.Count];
        // the landmark points are relative to the entire camera view. Need to convert each
        // to be in the reference frame of our bounding box
        for (int i = 0; i < face.Count; i++)
        {
            // face_landmarks[i] = new Vector2(face[i].X + (x1 + MARGIN), canvasHeight - (face[i].Y + (y1 + MARGIN)));
            face_landmarks[i] = new Vector3(face[i].X + (x1 + MARGIN), canvasHeight - (face[i].Y + (y1 + MARGIN)));
        }

        // TODO for Ben: Look into this but I don't think this is a problem in my case
        // the canvas is located at `center`, but the mask is located at the center of the bounding box
        // so need to adjust the anchor
        // set the pivot point of the canvas so that it rotates around the given center
        // var pivot = new Vector2(
        //     (float)(center.x - bounding_box[0] + MARGIN) / canvasWidth,
        //     1 - (float)(center.y - bounding_box[1] + MARGIN) / canvasHeight
        // );
        // rawImage.rectTransform.pivot = pivot;
        // rawImage.rectTransform.anchoredPosition = Vector2.zero;

        // set the scale ratio of the mask
        float scaleX = (float)boxWidth / height;
        float scaleY = (float)boxHeight / height;
        rawImage.transform.localScale = new Vector3(scaleX, scaleY, 1);

        // scale by distance * magic number
        float magicNumber = 0.75f;
        rawImage.transform.localScale *= distance * magicNumber;

        // We will pass in the specific edge list for each part of the face mask:
            //    1. Face Tesselation Mask (the main face mask mesh across the face)
            //    2. Face Contour Mask (the edges that go around the edge of the face)
            //    3. Right Brow
            //    4. Left Brow
            //    5. Right Eye
            //    6. Left Eye
            //    7. Right Iris
            //    8. Left Iris

        // var tesselation_cwidth = contourWidth;
        // var contour_cwidth = contourWidth;
        // var irises_cwidth = contourWidth;
        // var nose_cwidth = contourWidth;

        // Get the edge thicknesses for each part of the face mask
        var tesselation_cwidth = data.ContourThicknesses[index].FaceMeshTesselationThickness;
        var contour_cwidth = data.ContourThicknesses[index].FaceMeshContourThickness;
        var irises_cwidth = data.ContourThicknesses[index].FaceMeshLeftEyeThickness;  // TODO for Ben: Seperate out CONTOUR if want more customization

        // Get the color for each part of the face mask
        var tesselation_color_values = data.FaceMeshColors[index].FaceMeshTesselationColor;
        var contour_color_values = data.FaceMeshColors[index].FaceMeshContourColor;
        var irises_color_values = data.FaceMeshColors[index].FaceMeshLeftEyeColor;  // TODO for Ben: Se   erate out CONTOUR if want more customization

        Color  tesselation_color = new Color(tesselation_color_values.R, tesselation_color_values.G, tesselation_color_values.B, tesselation_color_values.A);
        Color contour_color = new Color(contour_color_values.R, contour_color_values.G, contour_color_values.B, contour_color_values.A);
        Color irises_color = new Color(irises_color_values.R, irises_color_values.G, irises_color_values.B, irises_color_values.A);


        // This is for if defining contour thicknesses is optional, but I've made it required
        // in the pipeline for face mesh
        // var thicknesses = data.ContourThicknesses;
        // if (thicknesses.Count > index) {
        //     // If we specified a specific countour thickness for this face, then use that instead
        //     var remoteWidth = thicknesses[index];
        //     if (remoteWidth > 0) {
        //         cwidth = remoteWidth;
        //     }
        // }

        // DrawLines takes a list of 3D points. We have each connection/edge mapping for each part of
        // our face mask stored as a list of indices (see FaceMeshConnections.cs). We need to convert
        // from List[Tuple[index, index]] List[Tuple[3dPoint, 3dPoint]]

        // Will define the pixels based on our landmarks (i.e. landmarks --> specific pixels --> draw them)

        // 1. Draw the edges of the mask
        DrawConnections(pixels, face_landmarks, FaceMeshConnections.FACEMESH_TESSELATION, center, tesselation_cwidth, tesselation_color, canvasWidth, canvasHeight);
        DrawConnections(pixels, face_landmarks, FaceMeshConnections.FACEMESH_CONTOURS, center, contour_cwidth, contour_color, canvasWidth, canvasHeight);
        DrawConnections(pixels, face_landmarks, FaceMeshConnections.FACEMESH_IRISES, center, irises_cwidth, irises_color, canvasWidth, canvasHeight);

        // 2. Draw the points themselves


        // TODO for Ben: Save Options for later
        // if (renderMode == RenderMode.Solid || renderMode == RenderMode.FlashSolid)
        // {
            // fill the mask
            // ScanLineFill(pixels, BuildEdgeList(face_landmarks), color, canvasWidth, canvasHeight);
            // ScanLineFill(pixels, FACEMESH_TESSELATION, color, canvasWidth, canvasHeight);
        // }
        // else if (renderMode == RenderMode.Outline || renderMode == RenderMode.FlashOutline)
        // {
        //     var cwidth = contourWidth;
        //     var thicknesses = data.ContourThicknesses;
        //     if (thicknesses.Count > index) {
        //         var remoteWidth = thicknesses[index];
        //         if (remoteWidth > 0) {
        //             cwidth = remoteWidth;
        //         }
        //     }
        //     // draw the contour
        //     DrawContour(pixels, mask_contour, center, cwidth, color, canvasWidth, canvasHeight);
        // }
        // else
        // {
        //     throw new System.ArgumentException("Invalid render mode: " + renderMode);
        // }

        texture.LoadRawTextureData(pixels);
        texture.Apply();
        rawImage.texture = texture;
    }

    List<Edge> BuildEdgeList(Vector2[] points)
    {
        List<Edge> edges = new List<Edge>();

        for (int i = 0; i < points.Length; i++)
        {
            Vector2 start = points[i];
            Vector2 end = points[(i + 1) % points.Length];

            // ignore horizontal edges
            if (start.y == end.y)
            {
                continue;
            }
            edges.Add(new Edge(start, end));
        }

        return edges;
    }

    void ScanLineFill(byte[] pixels, List<Edge> edges, Color color, int canvasWidth, int canvasHeight)
    {
        // color transparency maximum is 0.5f
        var fillColor = new Color(color.r, color.g, color.b, Mathf.Min(color.a, 0.5f));

        edges.Sort((a, b) => a.yMin.CompareTo(b.yMin));

        // prevent yStart from exceeding the height of the texture
        int yStart = Mathf.FloorToInt(edges[0].yMin);
        yStart = Mathf.Max(yStart, 0);
        
        // prevent yEnd from exceeding the height of the texture
        int yEnd = Mathf.CeilToInt(edges[edges.Count - 1].yMax);
        yEnd = Mathf.Min(yEnd, canvasHeight-1);

        List<Edge> activeEdges = new List<Edge>();

        for (int y = yStart; y < yEnd; y++)
        {
            // remove edges that end at this y
            activeEdges.RemoveAll(e => Mathf.CeilToInt(e.yMax) == y);

            // add edges that start at this y
            activeEdges.AddRange(edges.FindAll(e => e.yMin == y));

            // sort edges by x
            activeEdges.Sort((a, b) => a.x.CompareTo(b.x));

            for (int i = 0; i < activeEdges.Count; i += 2)
            {
                // fill the pixels between the pairs of edges

                // prevent xStart from exceeding the width of the texture
                int xStart = Mathf.CeilToInt(activeEdges[i].x);
                xStart = Mathf.Max(xStart, 0);

                // prevent xEnd from exceeding the width of the texture
                int xEnd = Mathf.FloorToInt(activeEdges[i + 1].x);
                xEnd = Mathf.Min(xEnd, canvasWidth-1);

                for (int x = xStart; x < xEnd; x++)
                {
                    SetPixelColor(pixels, x, y, fillColor, canvasWidth, canvasHeight);
                }
            }

            // update x for active edges
            for (int i = 0; i < activeEdges.Count; i++)
            {
                activeEdges[i].x += activeEdges[i].mInverse;
            }
        }
    }

    void DrawBorder(Texture2D texture, int width, int height, Color color)
    {
        for (int i = 0; i < width; i++)
        {
            texture.SetPixel(i, 0, color);
            texture.SetPixel(i, height - 1, color);
        }

        for (int i = 0; i < height; i++)
        {
            texture.SetPixel(0, i, color);
            texture.SetPixel(width - 1, i, color);
        }
    }

    // Take in a list of edges (a tuple of 3d points) and draw a line between each edge
    // in the specified style (color, thickness, pattern, etc). Update our pixel map
    // so that these drawn edges show on the canvas.
    //
    // including:
    // - pixels: the value of each pixel in our canvas.
    // - points: the (x,y,z) coordinates of each point in our graph
    // - connectionsMapping: a list of tuples, where each tuple represents a connection between two points.
    //                       Each point is represented by its index in the points list.
    // - center: The geometry center of the canvas
    // - width: the width of the line
    // - color: the color of the object
    // - canvasWidth: the width of the canvas that we're drawing on
    // - canvasHeight: the height of the canvas that we're drawing on
    void DrawConnections(byte[] pixels, Vector3[] points, HashSet<(int, int)> connectionsMapping, Vector2 center, int width, Color color, int canvasWidth, int canvasHeight) {
        // for (int i = 0; i < points.Length; i++)
        // {
        //     Vector3 start = points[i];
        //     Vector3 end = points[(i + 1) % points.Length];
            // DrawLine(pixels, start, end, center, width, color, canvasWidth, canvasHeight);
        // }

        // Iterate through each connection
        foreach (var connection in connectionsMapping) {
            int index1 = connection.Item1;
            int index2 = connection.Item2;

            // Access the points using the indices
            Vector3 start = points[index1];
            Vector3 end = points[index2];

            // For each connection, make sure that both of the landmarks
            // are visible and present (presence) enough to be drawn. If they aren't,
            // Then don't draw the connection 

            DrawLine(pixels, start, end, center, width, color, canvasWidth, canvasHeight);
        }
    }

    void DrawLines(byte[] pixels, Vector2[] points, Vector2 center, int width, Color color, int canvasWidth, int canvasHeight)
    {
        for (int i = 0; i < points.Length; i++)
        {
            Vector2 start = points[i];
            Vector2 end = points[(i + 1) % points.Length];
            DrawLine(pixels, start, end, center, width, color, canvasWidth, canvasHeight);
        }
    }

    // Draw a line between the start and end point
    // in the specified style (color, thickness, pattern, etc). Update our pixel map
    // so that this drawn line shows on the canvas.
    //
    // including:
    // - pixels: the value of each pixel in our canvas.
    // - start: the (x,y,z) coordinates of the start point of the line
    // - end: the (x,y,z) coordinates of the end point of the line
    // - center: The geometry center of the canvas
    // - width: the width of the line
    // - color: the color of the object
    // - canvasWidth: the width of the canvas that we're drawing on
    // - canvasHeight: the height of the canvas that we're drawing on
    // void DrawLine(byte[] pixels, Vector2 start, Vector2 end, Vector2 center, int width, Color color, int canvasWidth, int canvasHeight)
    void DrawLine(byte[] pixels, Vector3 start, Vector3 end, Vector2 center, int width, Color color, int canvasWidth, int canvasHeight)
    {
        int x0 = (int)start.x;
        int y0 = (int)start.y;
        int x1 = (int)end.x;
        int y1 = (int)end.y;

        int dx = Mathf.Abs(x1 - x0);
        int dy = Mathf.Abs(y1 - y0);

        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;

        int err = dx - dy;
        int err2;

        while (true) {
            DrawThickPoint(pixels, x0, y0, center, width, color, canvasWidth, canvasHeight);

            if (x0 == x1 && y0 == y1) {
                break;
            }

            err2 = 2 * err;

            if (err2 > -dy) {
                err -= dy;
                x0 += sx;
            }

            if (err2 < dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

    void DrawThickPoint(byte[] pixels, int x, int y, Vector2 center, int width, Color color, int canvasWidth, int canvasHeight)
    {
        int halfWidth = width / 2;
        int neg_x = halfWidth, pos_x = halfWidth, neg_y = halfWidth, pos_y = halfWidth;

        if (width % 2 == 0)
        {
            if (x >= center.x)
                neg_x = halfWidth - 1;
            else
                pos_x = halfWidth - 1;
            
            if (y >= center.y)
                neg_y = halfWidth - 1;
            else
                pos_y = halfWidth - 1;
        }

        for (int i = x-neg_x; i <= x+pos_x; i++)
        {
            for (int j = y-neg_y; j <= y+pos_y; j++)
            {
                if (i >= 0 && i < canvasWidth && j >= 0 && j < canvasHeight)
                {
                    SetPixelColor(pixels, i, j, color, canvasWidth, canvasHeight);
                }
            }
        }
    }


    void SetPixelColor(byte[] pixels, int x, int y, Color color, int width, int height)
    {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            // RGBA (4 bytes per pixel
            int index = (y * width + x) * 4;
            pixels[index] = (byte)(color.r * 255);
            pixels[index + 1] = (byte)(color.g * 255);
            pixels[index + 2] = (byte)(color.b * 255);
            pixels[index + 3] = (byte)(color.a * 255);
        } else {
            throw new System.ArgumentException("Invalid pixel coordinate: (" + x + ", " + y + ")" + " for texture size: " + width + "x" + height);
        }
    }

    void Update()
    {
        // if it's flashing type
        if (renderMode == RenderMode.FlashSolid || renderMode == RenderMode.FlashOutline)
        {
            // flash the object
            float alpha = Mathf.Abs(Mathf.Sin(Time.time * 2));
            rawImage.color = new Color(1, 1, 1, alpha);
        }
    }

    void OnDestroy()
    {
        if (texture != null)
        {
            Destroy(texture);
            texture = null; // just to be safe
        }
    }
}

class Edge
{
    // mInverse <-- inverse of slope m
    public float yMin, yMax, x, mInverse;

    public Edge(Vector2 start, Vector2 end)
    {
        if (start.y < end.y)
        {
            yMin = start.y;
            yMax = end.y;
            x = start.x;
        }
        else
        {
            yMin = end.y;
            yMax = start.y;
            x = end.x;
        }
        
        mInverse = (end.x - start.x) / (end.y - start.y);
    }
};

