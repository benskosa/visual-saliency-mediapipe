using UnityEngine;
using System.Collections.Generic;

namespace Assets.Scripts.DataTypes.FaceMesh {
    public class FaceMeshColors {
        public Color FaceMeshTesselationColor  { get; set; }
        public Color FaceMeshContourColor { get; set; }
        public Color FaceMeshRightBrowColor { get; set; }
        public Color FaceMeshLeftBrowColor { get; set; }
        public Color FaceMeshRightEyeColor { get; set; }
        public Color FaceMeshLeftEyeColor { get; set; }
        public Color FaceMeshRightIrisColor { get; set; }
        public Color FaceMeshLeftIrisColor { get; set; }


        // Default constructor
        public FaceMeshColors() { 
            FaceMeshTesselationColor = Color.yellow;
            FaceMeshContourColor = Color.yellow;
            FaceMeshRightBrowColor = Color.yellow;
            FaceMeshLeftBrowColor = Color.yellow;
            FaceMeshRightEyeColor = Color.yellow;
            FaceMeshLeftEyeColor = Color.yellow;
            FaceMeshRightIrisColor = Color.yellow;
            FaceMeshLeftIrisColor = Color.yellow;
        }

        public FaceMeshColors(Color faceMeshTesselationColor,
                              Color faceMeshContourColor,
                              Color faceMeshRightBrowColor,
                              Color faceMeshLeftBrowColor,
                              Color faceMeshRightEyeColor,
                              Color faceMeshLeftEyeColor,
                              Color faceMeshRightIrisColor,
                              Color faceMeshLeftIrisColor)
        {
            FaceMeshTesselationColor = faceMeshTesselationColor;
            FaceMeshContourColor = faceMeshContourColor;
            FaceMeshRightBrowColor = faceMeshRightBrowColor;
            FaceMeshLeftBrowColor = faceMeshLeftBrowColor;
            FaceMeshRightEyeColor = faceMeshRightEyeColor;
            FaceMeshLeftEyeColor = faceMeshLeftEyeColor;
            FaceMeshRightIrisColor = faceMeshRightIrisColor;
            FaceMeshLeftIrisColor = faceMeshLeftIrisColor;
        }

    }
}