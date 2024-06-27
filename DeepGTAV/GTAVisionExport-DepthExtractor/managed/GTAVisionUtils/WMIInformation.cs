using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Management;
using System.Collections.Generic;

namespace GTAVisionUtils
{

    public class WMIGraphicsInformation
    {
        public string deviceId;
        public string AdapterCompat;
        public string AdapterDACType;
        public string AdapterRAM;
        public int Availability;
        public string Caption;
        public string Description;
        public DateTime DriverDate;
        public string DriverVersion;
        public string PnPDeviceId;
        public string name;
        public string VideoArch;
        public string MemType;
        public string VideoProcessor;
        public string bpp;
        public string hrez;
        public string vrez;
        public string num_colors;
        public string cols;
        public string rows;
        public string refresh;
        public string scanMode;
        public string videoModeDesc;

        public WMIGraphicsInformation(ManagementBaseObject from)
        {
            deviceId = from.GetPropertyValue("DeviceID") as string;
            AdapterCompat = from.GetPropertyValue("AdapterCompatibility") as string;
            AdapterDACType = from.GetPropertyValue("AdapterDACType") as string;
            AdapterRAM = from.GetPropertyValue("AdapterRAM") as string;
            Caption = from.GetPropertyValue("Caption") as string;
            DriverVersion = from.GetPropertyValue("DriverVersion") as string;
            DriverDate = ManagementDateTimeConverter.ToDateTime( from.GetPropertyValue("DriverDate") as string);
            VideoProcessor = from.GetPropertyValue("VideoProcessor") as string;
            name = from.GetPropertyValue("Name") as string;

        }
    }
    public class WMIInformation
    {
        public Guid system_uuid;
        public string vendor;
        public string dnshostname;
        public string username;
        public string systemtype;
        public UInt64 totalmem;
        public List<WMIGraphicsInformation> gfxCards;
        /// <summary>
        /// gets wmi info for the current computer
        /// </summary>
        public WMIInformation()
        {
            var scope = new ManagementScope("ROOT\\CIMV2");
            var genQuery = new ObjectQuery("SELECT * FROM Win32_ComputerSystem");
            var result = new ManagementObjectSearcher(scope, genQuery).Get().Cast<ManagementBaseObject>();
            dnshostname = result.First().GetPropertyValue("DNSHostName") as string;
            username = result.First().GetPropertyValue("UserName") as string;
            if (username == null)
            {
                username = Environment.UserName;
            }
            systemtype = result.First().GetPropertyValue("SystemType") as string;
            totalmem = (UInt64) result.First().GetPropertyValue("TotalPhysicalMemory");
            var prodQuery = new ObjectQuery("SELECT * FROM Win32_ComputerSystemProduct");
            result = new ManagementObjectSearcher(scope, prodQuery).Get().Cast<ManagementBaseObject>();
            system_uuid = Guid.Parse( result.First().GetPropertyValue("UUID") as string);
            vendor = result.First().GetPropertyValue("Vendor") as string;

            var videoQuery = new ObjectQuery("SELECT * FROM Win32_VideoController");
            result = new ManagementObjectSearcher(scope, videoQuery).Get().Cast<ManagementBaseObject>();
            gfxCards = new List<WMIGraphicsInformation>();
            foreach (var obj in result)
            {
                gfxCards.Add(new WMIGraphicsInformation(obj));
            }

        }
    }
}
