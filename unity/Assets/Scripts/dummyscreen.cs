using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class dummyscreen : MonoBehaviour
{
    Camera mainCamera;
    // Start is called before the first frame update
    public Canvas dummyCanvas;
    Image dummyImage;
    void Start()
    {
        mainCamera = Camera.main;

        int dummy_pic_width = 640;
        int dummy_pic_height = 360;

        // put the four corners of the screen in the world
        Vector2[] corners = new Vector2[4];
        corners[0] = new Vector2(0, 0);
        corners[1] = new Vector2(dummy_pic_width, 0);
        corners[2] = new Vector2(dummy_pic_width, dummy_pic_height);
        corners[3] = new Vector2(0, dummy_pic_height);

        Vector3[] screenPos = new Vector3[4];
        for (int i = 0; i < 4; i++)
        {
            // hardcode: +-0.58, +-0.58, -0.58
            var worldSpaceDir = new Vector3(
                corners[i].x == 0 ? -0.58f : 0.58f,
                corners[i].y == 0 ? 0.58f : -0.58f,
                -0.58f
            );
            // when in editor, put it backwards, otherwise forward
            var worldSpacePos = mainCamera.transform.position - worldSpaceDir;
            screenPos[i] = worldSpacePos;
        }

        // place the dummy canvas in the world
        Vector3 center = (screenPos[0] + screenPos[1] + screenPos[2] + screenPos[3]) / 4;
        dummyCanvas.transform.position = center;

        // get the dummy image
        float width = Vector3.Distance(screenPos[0], screenPos[1]);
        float height = Vector3.Distance(screenPos[0], screenPos[3]);
        dummyImage = dummyCanvas.GetComponentInChildren<Image>();
        dummyImage.transform.localScale = new Vector3(width / height, 1, 1);

        // set the dummy to be perpendicular to the camera
        var forward = mainCamera.transform.forward;
        transform.forward = forward;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
