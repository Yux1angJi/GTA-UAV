using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Net;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Npgsql;
using GTA;
using Npgsql.NameTranslation;
using System.Configuration;
using NpgsqlTypes;
using System.Reflection;
using IniParser;
namespace GTAVisionUtils {
    
    public class PostgresExport {

        public static void InitSQLTypes()
        {
            NpgsqlConnection.MapEnumGlobally<GTA.Weather>("weather", new NpgsqlNullNameTranslator());
            NpgsqlConnection.MapEnumGlobally<DetectionType>();
            NpgsqlConnection.MapEnumGlobally<DetectionClass>(pgName: "detection_class", nameTranslator: new NpgsqlNullNameTranslator());
        }
        
        public static NpgsqlConnection OpenConnection()
        {
            var parser = new FileIniDataParser();
            var location = AppDomain.CurrentDomain.BaseDirectory;
            var data = parser.ReadFile(Path.Combine(location, "GTAVision.ini"));

            //UI.Notify(ConfigurationManager.AppSettings["database_connection"]);
            var str = data["Database"]["ConnectionString"];
            
            var conn = new NpgsqlConnection(str);
            conn.Open();
            return conn;
        }

        public static Guid InsertSystemData(NpgsqlConnection conn)
        {
            var systemInfo = new WMIInformation();
            using (var cmd = new NpgsqlCommand())
            {
                cmd.Connection = conn;
                cmd.Parameters.AddWithValue("@system_uuid", systemInfo.system_uuid);
                cmd.Parameters.AddWithValue("@vendor", systemInfo.vendor);
                cmd.Parameters.AddWithValue("@dnshostname", systemInfo.dnshostname);
                cmd.Parameters.AddWithValue("@username", systemInfo.username);
                cmd.Parameters.AddWithValue("@systemtype", systemInfo.systemtype);
                cmd.Parameters.AddWithValue("@totalmem", NpgsqlDbType.Bigint, systemInfo.totalmem);
                cmd.CommandText =
                    "INSERT INTO systems (system_uuid, vendor, dnshostname, username, systemtype, totalmem) VALUES " +
                    "(@system_uuid, @vendor, @dnshostname, @username, @systemtype, @totalmem) ON CONFLICT(system_uuid) " +
                    "DO UPDATE SET system_uuid = EXCLUDED.system_uuid RETURNING system_uuid";
                return Guid.Parse(cmd.ExecuteScalar().ToString());
            }
        }
        
        public static int InsertInstanceData(NpgsqlConnection conn)
        {
            
            var instanceinfo = new InstanceData();
            using (var cmd = new NpgsqlCommand())
            {
                cmd.Connection = conn;
                
                cmd.Parameters.AddWithValue("@host", System.Environment.MachineName);
                cmd.Parameters.AddWithValue("@iid", DBNull.Value);
                cmd.Parameters.AddWithValue("@typ", instanceinfo.type);
                cmd.Parameters.AddWithValue("@pubhost", DBNull.Value);
                cmd.Parameters.AddWithValue("@amiid", DBNull.Value);
                
                if (instanceinfo.type != "LOCALHOST")
                {
                    cmd.Parameters.AddWithValue("@host", instanceinfo.hostname);
                    cmd.Parameters.AddWithValue("@iid", instanceinfo.instanceid);
                    cmd.Parameters.AddWithValue("@typ", instanceinfo.type);
                    cmd.Parameters.AddWithValue("@pubhost", instanceinfo.publichostname);
                    cmd.Parameters.AddWithValue("@amiid", instanceinfo.amiid);
                }
                cmd.CommandText =
                    "SELECT instance_id FROM instances WHERE hostname=@host AND instancetype=@typ AND instanceid=@iid AND amiid=@amiid AND publichostname=@pubhost";
                var id = cmd.ExecuteScalar();
                if (id == null)
                {
                    cmd.CommandText =
                    "INSERT INTO instances (hostname, instanceid, instancetype, publichostname, amiid) VALUES (@host, @iid, @typ, @pubhost, @amiid) " +
                    "RETURNING instance_id";
                    return (int) cmd.ExecuteScalar();
                }

                return (int) id;

            }
        }

        public static async Task<int> StartSession(string name)
        {
            return await Task.Run(() => StartSessionImpl(name));
        }
        public static int StartSessionImpl(string name)
        {
            var conn = OpenConnection();
            int result = 0;
            //int instance = InsertInstanceData(conn);
            using (var cmd = new NpgsqlCommand())
            {
                
                cmd.Connection = conn;
                cmd.CommandText = "INSERT INTO sessions (name, start) VALUES (@name, @start) ON CONFLICT DO NOTHING";
                cmd.Parameters.AddWithValue("@name", name);
                cmd.Parameters.AddWithValue("@start", DateTime.UtcNow);
                cmd.ExecuteNonQuery();
                cmd.CommandText = "SELECT session_id FROM sessions WHERE name = @name";
                result = (int)cmd.ExecuteScalar();
            }
            conn.Close();
            return result;
        }

        public static async Task StopSession(int sessionid)
        {
            await Task.Run(() => StopSessionImpl(sessionid));
        }
        public static void StopSessionImpl(int sessionid)
        {
            var conn = OpenConnection();
            using (var cmd = new NpgsqlCommand())
            {
                cmd.Connection = conn;
                cmd.CommandText = @"UPDATE sessions SET ""end"" = @endtime WHERE session_id = @sessionid";
                cmd.Parameters.AddWithValue("@endtime", DateTime.UtcNow);
                cmd.Parameters.AddWithValue("@sessionid", sessionid);
                cmd.ExecuteNonQuery();
            }
            conn.Close();
        }

        public static async Task<GTARun> StartRun(int sessionid)
        {
            var t = Task.Run(() => StartRunImpl(sessionid));
            return await t;
        }
        public static GTARun StartRunImpl(int sessionid)
        {
            var run = new GTARun();
            run.guid = Guid.NewGuid();
            run.archiveKey = Path.Combine("images", run.guid + ".zip");
            var conn = OpenConnection();
            int instanceid = InsertInstanceData(conn);
            InsertSystemData(conn);
            using (var cmd = new NpgsqlCommand())
            {
                cmd.Connection = conn;
                cmd.CommandText = "INSERT INTO runs (runguid, archivepath, session_id, instance_id) VALUES (@guid, @archivekey, @session, @instance);";
                cmd.Parameters.AddWithValue("@guid", run.guid);
                cmd.Parameters.AddWithValue("@archivekey", run.archiveKey);
                cmd.Parameters.AddWithValue("@session", sessionid);
                cmd.Parameters.AddWithValue("@instance", instanceid);
                cmd.ExecuteNonQuery();
            }
            conn.Close();
            
            return run;
        }

        public static void StopRun(GTARun run)
        {
        }

        public static async void SaveSnapshot(GTAData data, Guid runId)
        {
            await Task.Run(() => SaveSnapshotImpl(data, runId));
        }
        
        public static void SaveSnapshotImpl(GTAData data, Guid runId)
        {
            var conn = OpenConnection();
            var trans = conn.BeginTransaction();
            using (NpgsqlCommand cmd = new NpgsqlCommand())
            {
                
                cmd.Connection = conn;
                cmd.Transaction = trans;
                cmd.CommandText =
                    "INSERT INTO snapshots (run_id, version, imagepath, timestamp, timeofday, currentweather, camera_pos, camera_direction, camera_fov, view_matrix, proj_matrix, width, height) " +
                    "VALUES ( (SELECT run_id FROM runs WHERE runguid=@guid), " +
                    "@Version, @Imagepath, @Timestamp, @Timeofday, @currentweather, ST_MakePoint(@x, @y, @z), ST_MakePoint(@dirx, @diry, @dirz), @fov, @view_matrix, @proj_matrix, @width, @height ) " +
                    "RETURNING snapshot_id;";
                cmd.Parameters.Add(new NpgsqlParameter("@version", data.Version));
                cmd.Parameters.Add(new NpgsqlParameter("@imagepath", data.ImageName));
                cmd.Parameters.Add(new NpgsqlParameter("@timestamp", data.Timestamp));
                cmd.Parameters.Add(new NpgsqlParameter("@timeofday", data.LocalTime));
                cmd.Parameters.Add(new NpgsqlParameter("@currentweather", data.CurrentWeather));
                cmd.Parameters.Add(new NpgsqlParameter("@x", data.Pos.X));
                cmd.Parameters.Add(new NpgsqlParameter("@y", data.Pos.Y));
                cmd.Parameters.Add(new NpgsqlParameter("@z", data.Pos.Z));
                cmd.Parameters.AddWithValue("@dirx", data.CamDirection.X);
                cmd.Parameters.AddWithValue("@diry", data.CamDirection.Y);
                cmd.Parameters.AddWithValue("@dirz", data.CamDirection.Z);
                cmd.Parameters.AddWithValue("@fov", data.CamFOV);
                cmd.Parameters.AddWithValue("@view_matrix", data.ViewMatrix.ToArray());
                cmd.Parameters.AddWithValue("@proj_matrix", data.ProjectionMatrix.ToArray());
                cmd.Parameters.AddWithValue("@width", data.ImageWidth);
                cmd.Parameters.AddWithValue("@height", data.ImageHeight);
                cmd.Parameters.Add(new NpgsqlParameter("@guid", runId));
                int snapshotid = (int)cmd.ExecuteScalar();
                cmd.Parameters.Clear();
                cmd.CommandText =
                    "INSERT INTO snapshot_weathers (snapshot_id, weather_type, snapshot_page) VALUES (@snapshot, @weather, @page);";
                cmd.Parameters.AddWithValue("@snapshot", NpgsqlDbType.Integer, snapshotid);
                cmd.Parameters.AddWithValue("@weather", NpgsqlDbType.Enum, Weather.Unknown);
                cmd.Parameters.Add("@page", NpgsqlDbType.Integer);
                cmd.Prepare();
                for (int i = 0; i < data.CapturedWeathers.Count; ++i)
                {
                    cmd.Parameters["@weather"].Value = data.CapturedWeathers[i];
                    cmd.Parameters["@page"].Value = i;
                    cmd.ExecuteNonQuery();
                }

                cmd.Parameters.Clear();
                cmd.Parameters.Add("@snapshot", NpgsqlDbType.Integer);
                cmd.Parameters.AddWithValue("@type", NpgsqlDbType.Enum, DetectionType.background);
                cmd.Parameters.Add("@x", NpgsqlDbType.Real);
                cmd.Parameters.Add("@y", NpgsqlDbType.Real);
                cmd.Parameters.Add("@z", NpgsqlDbType.Real);
                cmd.Parameters.Add("@xrot", NpgsqlDbType.Real);
                cmd.Parameters.Add("@yrot", NpgsqlDbType.Real);
                cmd.Parameters.Add("@zrot", NpgsqlDbType.Real);
                cmd.Parameters.Add("@bbox", NpgsqlDbType.Box);
                cmd.Parameters.Add("@minx", NpgsqlDbType.Real);
                cmd.Parameters.Add("@miny", NpgsqlDbType.Real);
                cmd.Parameters.Add("@minz", NpgsqlDbType.Real);
                cmd.Parameters.Add("@maxx", NpgsqlDbType.Real);
                cmd.Parameters.Add("@maxy", NpgsqlDbType.Real);
                cmd.Parameters.Add("@maxz", NpgsqlDbType.Real);
                cmd.Parameters.AddWithValue("@class", NpgsqlDbType.Enum, DetectionClass.Unknown);
                cmd.Parameters.Add("@handle", NpgsqlDbType.Integer);
                cmd.CommandText =
                    "INSERT INTO detections (snapshot_id, type, pos, rot, bbox, class, handle, bbox3d) VALUES " +
                    "(@snapshot, @type, ST_MakePoint(@x,@y,@z), ST_MakePoint(@xrot, @yrot, @zrot), @bbox, @class, @handle," +
                    "ST_3DMakeBox(ST_MakePoint(@minx,@miny,@minz), ST_MakePoint(@maxx, @maxy, @maxz)));";
                cmd.Prepare();
                
                
                foreach (var detection in data.Detections) {
                    
                    cmd.Parameters["@snapshot"].Value = snapshotid;
                    cmd.Parameters["@type"].Value = detection.Type;
                    cmd.Parameters["@x"].Value = detection.Pos.X;
                    cmd.Parameters["@y"].Value = detection.Pos.Y;
                    cmd.Parameters["@z"].Value = detection.Pos.Z;
                    cmd.Parameters["@xrot"].Value = detection.Rot.X;
                    cmd.Parameters["@yrot"].Value = detection.Rot.Y;
                    cmd.Parameters["@zrot"].Value = detection.Rot.Z;
                    cmd.Parameters["@bbox"].Value =
                        new NpgsqlBox(detection.BBox.Max.Y, detection.BBox.Max.X, detection.BBox.Min.Y, detection.BBox.Min.X);
                    cmd.Parameters["@class"].Value = detection.cls;
                    cmd.Parameters["@handle"].Value = detection.Handle;
                    cmd.Parameters["@minx"].Value = detection.BBox3D.Minimum.X;
                    cmd.Parameters["@miny"].Value = detection.BBox3D.Minimum.Y;
                    cmd.Parameters["@minz"].Value = detection.BBox3D.Minimum.Z;

                    cmd.Parameters["@maxx"].Value = detection.BBox3D.Maximum.X;
                    cmd.Parameters["@maxy"].Value = detection.BBox3D.Maximum.Y;
                    cmd.Parameters["@maxz"].Value = detection.BBox3D.Maximum.Z;

                    cmd.ExecuteNonQuery();
                }
            }
            trans.Commit();
            conn.Close();
        }
    }

}
