using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GTA;
using GTA.Math;

namespace GTAVisionExport {
    public class StereoCamera
    {
        Camera leftCam;
        Camera rightCam;
        private Camera gameCam;
        public StereoCamera(Entity e)
        {
            leftCam = World.CreateCamera(new Vector3(1, 0, 0), Vector3.RelativeLeft, GameplayCamera.FieldOfView);
            rightCam = World.CreateCamera(new Vector3(-1, 0, 0), Vector3.RelativeFront, GameplayCamera.FieldOfView);
            gameCam = World.RenderingCamera;
            leftCam.AttachTo(e, new Vector3(0, 10, 0));
            rightCam.AttachTo(e, new Vector3(0, 10, 0));
        }

        public void ActivateLeft()
        {
            World.RenderingCamera = leftCam;
        }

        public void ActivateRight()
        {
            World.RenderingCamera = rightCam;
        }

        public void Deactivate()
        {
            World.RenderingCamera = gameCam;
        }
    }
}
