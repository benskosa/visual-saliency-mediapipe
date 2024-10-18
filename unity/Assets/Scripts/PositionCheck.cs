using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PositionCheck : DataProcessor
{
    public LocatableCamera locatableCamera = null;

    public override void ProcessData(RecognitionData data, uint width, uint height)
    {
        // if (data.PoseValid)
        // {
        //     var position = data.CameraPosition;
        //     var rotation = data.CameraRotation;

        //     MainThreadDispatcher.Instance.Enqueue(() =>
        //     {
        //         // set the position
        //         transform.position = new Vector3(position.X, position.Y, position.Z);

        //         Vector3 dir = new Vector3(rotation.X, rotation.Y, rotation.Z);
        //         transform.rotation = Quaternion.LookRotation(dir);
        //     });
        // }

        if (data.Timestamp > 0)
        {
            (Vector3 position, Quaternion rotation) = locatableCamera.GetPosRot(data.Timestamp);
            MainThreadDispatcher.Instance.Enqueue(() =>
            {
                // set the position
                transform.position = position;
                transform.rotation = rotation;
            });
        }
    }
}
