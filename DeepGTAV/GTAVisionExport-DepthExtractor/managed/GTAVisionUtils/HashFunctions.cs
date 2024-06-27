using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GTA.Math;
using GTA.Native;

namespace GTAVisionUtils
{
    public class HashFunctions
    {
        public static Vector2 Convert3dTo2d(Vector3 pos)
        {
            OutputArgument tmpResX = new OutputArgument();
            OutputArgument tmpResY = new OutputArgument();
            if (Function.Call<bool>(Hash._WORLD3D_TO_SCREEN2D, (InputArgument) pos.X,
                (InputArgument) pos.Y, (InputArgument) pos.Z,
                (InputArgument) tmpResX, (InputArgument) tmpResY))
            {
                Vector2 v2;
                v2.X = tmpResX.GetResult<float>();
                v2.Y = tmpResY.GetResult<float>();
                return v2;
            }
            return new Vector2(-1f, -1f);
        }
        public static void Draw3DLine(Vector3 iniPos, Vector3 finPos, byte col_r = 255, byte col_g = 255, byte col_b = 255, byte col_a = 255) {
            Function.Call(Hash.DRAW_LINE, new InputArgument[]
            {
                iniPos.X,
                iniPos.Y,
                iniPos.Z,
                finPos.X,
                finPos.Y,
                finPos.Z,
                (int)col_r,
                (int)col_g,
                (int)col_b,
                (int)col_a
            });
        }
        public static void DrawRect(float x, float y, float w, float h, byte r = 255, byte g = 255, byte b = 255, byte a = 255) {
            Function.Call(Hash.DRAW_RECT, new InputArgument[] {
                x, y,
                w, h,
                (int)r, (int)g, (int)b, (int)a
            });
        }
        
    }
}
