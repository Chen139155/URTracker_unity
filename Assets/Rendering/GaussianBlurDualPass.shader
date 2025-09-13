Shader "UI/UIGaussianBlur"
{
    Properties
    {
        _MainTex("Base (RGB)", 2D) = "white" {}
        _Size("Blur Size", Range(0, 30)) = 1
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            float _Size;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col = fixed4(0,0,0,0);

                // 高斯权重 (7个采样点)
                float weights[7] = {0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12};
                int halfLen = 3;

                // 从中心向四周采样
                for (int x=-halfLen; x<=halfLen; x++)
                {
                    for (int y=-halfLen; y<=halfLen; y++)
                    {
                        float weight = weights[x + halfLen] * weights[y + halfLen];
                        float2 offset = float2(x, y) * _MainTex_TexelSize.xy * _Size;
                        col += tex2D(_MainTex, i.uv + offset) * weight;
                    }
                }

                return col;
            }

            ENDCG
        }
    }

    FallBack Off
}
