using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GTA;
using GTA.Math;

namespace GTAVisionExport
{
    
    public class speedAndTime
    {
        public int gameTime { get; set; }
        public float speed { get; set; }
        public bool initialized { get; set; }
        public speedAndTime()
        {
            gameTime = 0;
            speed = 0;
            initialized = false;
        }
        public void setTime(int time, float speed)
        {
            this.gameTime = time;
            this.speed = speed;
            this.initialized = true;
        }
        public void clearTime()
        {
            this.initialized = false;
        }
        public Boolean checkTrafficJam(int time, float speed)
        {
            //UI.Notify("last time" + this.gameTime);
            //UI.Notify("time now" + time);
            if (!initialized)
            {
                this.gameTime = time;
                this.speed = speed;
                this.initialized = true;
                return false;
            }
            /* Game.Gametime is in ms, so 1000000 ms = 16.6 min*/
            else if (time >= this.gameTime + 200000)
            {
                return true;
            }
            else return false;
        }
    }

    public enum GameStatus
    {
        NeedReload,
        NeedStart,
        NoActionNeeded
    }

    public class GTAConst
    {
        public static Vector3 StartPos = new Vector3(311.7819f, -1372.574f, 31.84874f);
    }
}
