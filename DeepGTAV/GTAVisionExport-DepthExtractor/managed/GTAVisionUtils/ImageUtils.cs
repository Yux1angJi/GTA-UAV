using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitMiracle.LibTiff.Classic;

namespace GTAVisionUtils
{
    public class ImageUtils
    {
        private static Task imageTask;
        private static byte[] lastCapturedBytes;

        public static byte[] getLastCapturedFrame()
        {
            WaitForProcessing();
            return lastCapturedBytes;
        }

        public static void WaitForProcessing()
        {
            if (imageTask == null) return;
            imageTask.Wait();
        }

        public static void StartUploadTask(ZipArchive archive, string name, int w, int h,
            List<byte[]> colors, byte[] depth, byte[] stencil)
        {
            WaitForProcessing();
            imageTask = Task.Run(() => UploadToArchive(archive, name, w, h, colors, depth, stencil));
        }

        public static void UploadToArchive(ZipArchive archive, string name, int w, int h,
            List<byte[]> colors, byte[] depth, byte[] stencil)
        {
            var memstream = new MemoryStream();
            var tiff = Tiff.ClientOpen(name, "w", memstream, new TiffStream());
            WriteToTiff(tiff, w, h, colors, depth, stencil);
            tiff.Flush();
            tiff.Close();
            var entry = archive.CreateEntry(name + ".tiff", CompressionLevel.NoCompression);
            var entryStream = entry.Open();
            lastCapturedBytes = memstream.ToArray();
            entryStream.Write(lastCapturedBytes, 0, lastCapturedBytes.Length);
            entryStream.Close();
            //tiff.Close();
            memstream.Close();

        }

        public static void WriteToTiff(Tiff t, int width, int height, List<byte[]> colors, byte[] depth, byte[] stencil)
        {
            var pages = colors.Count + 2;
            var page = 0;
            foreach (var color in colors)
            {
                t.CreateDirectory();
                t.SetField(TiffTag.IMAGEWIDTH, width);
                t.SetField(TiffTag.IMAGELENGTH, height);
                t.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                t.SetField(TiffTag.SAMPLESPERPIXEL, 4);
                t.SetField(TiffTag.ROWSPERSTRIP, height);
                t.SetField(TiffTag.BITSPERSAMPLE, 8);
                t.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
                t.SetField(TiffTag.PHOTOMETRIC, Photometric.RGB);
                t.SetField(TiffTag.COMPRESSION, Compression.JPEG);
                t.SetField(TiffTag.JPEGQUALITY, 60);
                t.SetField(TiffTag.PREDICTOR, Predictor.HORIZONTAL);
                t.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.UINT);
                t.SetField(TiffTag.PAGENUMBER, page, pages);
                t.WriteEncodedStrip(0, color, color.Length);
                page++;
                t.WriteDirectory();
            }
            
            t.CreateDirectory();
            //page 2
            t.SetField(TiffTag.IMAGEWIDTH, width);
            t.SetField(TiffTag.IMAGELENGTH, height);
            t.SetField(TiffTag.ROWSPERSTRIP, height);
            t.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            t.SetField(TiffTag.SAMPLESPERPIXEL, 1);
            t.SetField(TiffTag.BITSPERSAMPLE, 32);
            t.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
            t.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
            t.SetField(TiffTag.COMPRESSION, Compression.LZW);
            t.SetField(TiffTag.PREDICTOR, Predictor.FLOATINGPOINT);
            t.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.IEEEFP);
            t.SetField(TiffTag.PAGENUMBER, page, pages);
            t.WriteEncodedStrip(0, depth, depth.Length);
            page++;
            t.WriteDirectory();

            t.SetField(TiffTag.IMAGEWIDTH, width);
            t.SetField(TiffTag.IMAGELENGTH, height);
            t.SetField(TiffTag.ROWSPERSTRIP, height);
            t.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            t.SetField(TiffTag.SAMPLESPERPIXEL, 1);
            t.SetField(TiffTag.BITSPERSAMPLE, 8);
            t.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
            t.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
            t.SetField(TiffTag.COMPRESSION, Compression.LZW);
            t.SetField(TiffTag.PREDICTOR, Predictor.HORIZONTAL);
            t.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.UINT);
            t.SetField(TiffTag.PAGENUMBER, page, pages);
            t.WriteEncodedStrip(0, stencil, stencil.Length);
            t.WriteDirectory();
            t.Flush();
        }
        
    }
}
