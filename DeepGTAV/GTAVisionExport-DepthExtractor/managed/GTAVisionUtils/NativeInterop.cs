using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SharpDX;

namespace GTAVisionUtils {
    [StructLayout(LayoutKind.Sequential)]
    public struct rage_matrices
    {
        public Matrix world;
        public Matrix worldView;
        public Matrix worldViewProjection;
        public Matrix invView;
    }
    public class VisionNative
    {
        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_depth_buffer", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetDepthBuffer(out IntPtr buf);

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_color_buffer", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetFrontBuffer(out IntPtr buf);

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_stencil_buffer", CharSet = CharSet.Unicode,
            CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetStencilBuffer(out IntPtr buf);

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_constant_buffer", CharSet = CharSet.Unicode,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetConstants(out rage_matrices buf);

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_last_depth_time",
             CallingConvention = CallingConvention.Cdecl)]
        public static extern long GetLastDepthTime();

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_last_color_time",
             CallingConvention = CallingConvention.Cdecl)]
        public static extern long GetLastColorTime();

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_last_constant_time",
             CallingConvention = CallingConvention.Cdecl)]
        public static extern long GetLastConstantTime();

        [DllImport("GTAVisionNative.asi", EntryPoint = "export_get_current_time",
             CallingConvention = CallingConvention.Cdecl)]
        public static extern long GetCurrentTime();

        public static byte[] GetDepthBuffer()
        {
            IntPtr buf;
            var sz = GetDepthBuffer(out buf);
            if (sz == -1) return null;
            var result = new byte[sz];
            Marshal.Copy(buf, result, 0, sz);
            return result;
        }

        public static byte[] GetColorBuffer()
        {
            IntPtr buf;
            var sz = GetFrontBuffer(out buf);
            if (sz == -1) return null;
            var result = new byte[sz];
            Marshal.Copy(buf, result, 0, sz);
            return result;
        }

        public static byte[] GetStencilBuffer()
        {
            IntPtr buf;
            var sz = GetStencilBuffer(out buf);
            if (sz == -1) return null;
            var result = new byte[sz];
            Marshal.Copy(buf, result, 0, sz);
            return result;
        }

        public static rage_matrices? GetConstants()
        {
            
            rage_matrices rv;
            rv.invView = Matrix.Identity;
            rv.world = Matrix.Identity;
            rv.worldView = Matrix.Identity;
            rv.worldViewProjection = Matrix.Identity;
            int res = GetConstants(out rv);
            if (res == -1) return null;
            return rv;
        }

    }
}
