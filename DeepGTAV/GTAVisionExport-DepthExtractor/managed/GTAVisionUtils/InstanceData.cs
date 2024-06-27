using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace GTAVisionUtils {
    public class InstanceData
    {
        public string amiid;
        public string hostname;
        public string instanceid;
        public string publichostname;
        public string type;

        public InstanceData()
        {
            hostname = Environment.MachineName;
            try {
                var req = WebRequest.Create("http://169.254.169.254/latest/meta-data");
                req.Timeout = 500;
                
                HttpWebResponse resp = req.GetResponse() as HttpWebResponse;
                resp.Close();
            } catch (WebException e)
            {
                type = "LOCALHOST";
            }

            if (type != "LOCALHOST") {
                StreamReader reader;
                var resp = WebRequest.Create("http://169.254.169.254/latest/meta-data/ami-id/").GetResponse() as HttpWebResponse;
                reader = new StreamReader(resp.GetResponseStream());
                amiid = reader.ReadToEnd();
                resp.Close();
                resp = WebRequest.Create("http://169.254.169.254/latest/meta-data/hostname/").GetResponse() as HttpWebResponse;
                reader = new StreamReader(resp.GetResponseStream());
                hostname = reader.ReadToEnd();
                resp.Close();
                resp = WebRequest.Create("http://169.254.169.254/latest/meta-data/instance-id/").GetResponse() as HttpWebResponse;
                reader = new StreamReader(resp.GetResponseStream());
                instanceid = reader.ReadToEnd();
                resp.Close();
                resp = WebRequest.Create("http://169.254.169.254/latest/meta-data/instance-type/").GetResponse() as HttpWebResponse;
                reader = new StreamReader(resp.GetResponseStream());
                type = reader.ReadToEnd();
                resp.Close();
                resp = WebRequest.Create("http://169.254.169.254/latest/meta-data/public-hostname/").GetResponse() as HttpWebResponse;
                reader = new StreamReader(resp.GetResponseStream());
                publichostname = reader.ReadToEnd();
                resp.Close();


            }
            
        }
    }
}
