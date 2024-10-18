Shader "Custom/UnlitTransparentColor"
{
    Properties
    {
        _Color ("Main Color", Color) = (1, 1, 1, 1)
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            ZWrite Off
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata_t
            {
                float4 vertex : POSITION;

                UNITY_VERTEX_INPUT_INSTANCE_ID // Required when using instancing
            };

            struct v2f
            {
                float4 pos : SV_POSITION;

                UNITY_VERTEX_OUTPUT_STEREO // Required when using multi-view or single-pass stereo
            };

            float4 _Color;

            v2f vert (appdata_t v)
            {
                v2f o;

                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.pos = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return _Color;
            }
            ENDCG
        }
    }
}
