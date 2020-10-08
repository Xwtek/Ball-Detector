using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetection
{
    class Program
    {
        private static int Threshold = 50;
        private static int ErodeIterations = 3;
        private static int DilateIterations = 3;

        private const string BgFrameWindowName = "Background Frame";
        private const string RawFrameWindowName = "Raw Frame";
        private const string GrayscaleDifferenceFrameWindowName = "Grayscale Difference Frame";
        private const string BinaryDifferenceFrameWindowName = "Binary Difference Frame";
        private const string DenoisedDifferenceFrameWindowName = "Denoised Difference Frame";
        private const string FinalFrameWindowName = "Final Frame";

        private static Mat bgFrame = new Mat();
        private static Mat rawFrame = new Mat();
        private static Mat diffFrame = new Mat();
        private static Mat grayDiffFrame = new Mat();
        private static Mat binDiffFrame = new Mat();
        private static Mat denoisedDiffFrame = new Mat();
        private static Mat finalFrame = new Mat();

        private static MCvScalar drawingColor = new Bgr(Color.Red).MCvScalar;
        static void Main(string[] args)
        {
            string videofile;
            if (args.Length > 0)
            {
                videofile = args[0];
            }
            else
            {
                Console.Write("Enter video file: ");
                videofile = Console.ReadLine();
            }
            using (var capture = new VideoCapture(videofile))
            {
                if (capture.IsOpened)
                {
                    Console.WriteLine($"{videofile} is opened");
                    Console.WriteLine("Press ESCAPE key to exit");
                    Console.WriteLine("Press any other key to go to the next frame");

                    bgFrame = capture.QueryFrame();
                    CvInvoke.Imshow(BgFrameWindowName, bgFrame);

                    VideoProcessingLoop(capture, bgFrame);
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Unable to open {videofile}");
                    Console.ReadKey();
                }
            }
        }

        private static void VideoProcessingLoop(VideoCapture capture, Mat bgFrame)
        {
            var stopwatch = new Stopwatch();

            int framenum = 1;
            while (true)
            {
                rawFrame = capture.QueryFrame();
                if (rawFrame != null)
                {
                    framenum++;
                    stopwatch.Restart();
                    ProcessFrame(bgFrame, Threshold, ErodeIterations, DilateIterations);
                    stopwatch.Stop();
                    WriteFrameInfo(stopwatch.ElapsedMilliseconds, framenum);
                    ShowWindowsWithImageProcessingStages();

                    int key = CvInvoke.WaitKey(0);
                    if (key == 27) return;
                }
                else
                {
                    capture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames, 0);
                    framenum = 0;
                }
            }
        }

        private static void ShowWindowsWithImageProcessingStages()
        {
            CvInvoke.Imshow(RawFrameWindowName, rawFrame);
            CvInvoke.Imshow(GrayscaleDifferenceFrameWindowName, grayDiffFrame);
            CvInvoke.Imshow(BinaryDifferenceFrameWindowName, binDiffFrame);
            CvInvoke.Imshow(DenoisedDifferenceFrameWindowName, denoisedDiffFrame);
            CvInvoke.Imshow(FinalFrameWindowName, finalFrame);
        }

        private static void WriteFrameInfo(long elapsedMilliseconds, int framenum)
        {
            WriteMultilineText(finalFrame, new string[]
            {
                $"Frame Number: {framenum}",
                $"Processing Time: {elapsedMilliseconds} ms"
            }, new Point(5, 10));
        }

        private static void WriteMultilineText(Mat finalFrame, string[] v, Point point)
        {
            CvInvoke.PutText(finalFrame, string.Join("\n", v), point, FontFace.HersheyPlain, 12, drawingColor);
        }

        private static void ProcessFrame(Mat bgFrame, int threshold, int erodeIterations, int dilateIterations)
        {
            CvInvoke.AbsDiff(bgFrame, rawFrame, diffFrame);
            CvInvoke.CvtColor(diffFrame, grayDiffFrame, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(grayDiffFrame, binDiffFrame, threshold, 255, ThresholdType.Binary);
            CvInvoke.Erode(binDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), erodeIterations, BorderType.Default, new MCvScalar(1));
            CvInvoke.Dilate(denoisedDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), erodeIterations, BorderType.Default, new MCvScalar(1));

            rawFrame.CopyTo(finalFrame);
            DetectObject(denoisedDiffFrame, finalFrame);
        }

        private static void DetectObject(Mat detectionFrame, Mat displayFrame)
        {
            using (var contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(detectionFrame, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                if (contours.Size > 0)
                {
                    var chosen = (0d, 0);
                    for (int i = 0; i < contours.Size; i++)
                    {
                        chosen = Max(chosen, (CvInvoke.ContourArea(contours[i]), i));
                    }
                    MarkDetectedObject(displayFrame, contours[chosen.Item2], chosen.Item1);
                }
            }
        }

        private static void MarkDetectedObject(Mat frame, VectorOfPoint contour, double area)
        {
            var box = CvInvoke.BoundingRectangle(contour);
            CvInvoke.Polylines(frame, contour, true, drawingColor);
            CvInvoke.Rectangle(frame, box, drawingColor);
            var center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);
            WriteMultilineText(frame, new string[]
            {
                $"Area: {area}",
                $"Position: ({center.X}, {center.Y})"
            }, new Point(box.Right + 5, center.Y));
        }

        private static T Max<T>(T a, T b) where T : IComparable<T>
        {
            if (a.CompareTo(b) < 0) return b;
            return a;
        }
        private static T Min<T>(T a, T b) where T : IComparable<T>
        {
            if (a.CompareTo(b) > 0) return b;
            return a;
        }
    }
}
