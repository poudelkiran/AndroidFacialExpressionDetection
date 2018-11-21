package com.example.kiran.FacialExpression;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.PermissionChecker;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.opencv.imgproc.Imgproc;

import static android.Manifest.*;
public class camera extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    public int cameraId = 0;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    //Our classifier has input image size 48*48
    private static final int image_size = 48;
    private CascadeClassifier cascadeClassifier;
    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private ImageClassifier classifier = new ImageClassifier();
    TensorFlowInferenceInterface tf;
    private ImageProcess imageProcess;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    //Load OpenCV and initialize OpenCV dependencies
                    initializeOpenCVDependencies();/**/
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    private void initializeOpenCVDependencies() {
        try {
            //Load Resource file 'haar_cascade_frontal_default.xml' that identifies the frontal face
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            cascadeClassifier.load( mCascadeFile.getAbsolutePath() );
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
    }
    public camera() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final Mat src = inputFrame.rgba();
        Mat image = src;
        Core.flip(src, image, 0);
        Mat display_image = inputFrame.rgba();
        display_image=image;
        // Create a grayscale image
        Imgproc.cvtColor(image, grayscaleImage, Imgproc.COLOR_RGBA2RGB);
        final MatOfRect faces = new MatOfRect();
        // Use the haar_casscade classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }
        // If there are any faces found, draw a rectangle around it
        final Rect[] facesArray = faces.toArray();
        for (int i = 0; i <facesArray.length; i++) {
            //Initialize the HashMap that has color codes
            Map<String, Scalar> map = new HashMap<String, Scalar>();
            map.put("Angry", new Scalar(255,0,0));
            map.put("Disgust", new Scalar(0,255, 0));
            map.put("Fear", new Scalar(0,0,0));
            map.put("Happy", new Scalar(255,255,0));
            map.put("Sad", new Scalar(0,0,255));
            map.put("Surprise", new Scalar(0,255,255));
            map.put("Neutral", new Scalar(211,211,211));
            //Crop the faces of the image
            //tl and br contains the points of top left and bottom right of the found face respectively.
            Mat cropped_face = src.submat((int) facesArray[i].tl().y, (int) facesArray[i].br().y, (int) facesArray[i].tl().x, (int) facesArray[i].br().x);
            image = cropped_face;
            //Now the image contains the cropped_face. Lets convert the faces into the graysvale
            Imgproc.cvtColor(image, grayscaleImage, Imgproc.COLOR_RGB2GRAY);
            //Create a new bitmap of size image
            final Bitmap bmp = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            //Convert image from Mat to Bitmap
            Utils.matToBitmap(grayscaleImage, bmp);
            final Mat finalImage = image;
            //Crop Image to the image_size. It retubrns image of size image_size*imagesize
            Bitmap croppedBitmap = imageProcess.CropBitmap(bmp, image_size);
            Mat mat = new Mat();
            //Convert bitmap to mat
            Utils.bitmapToMat(croppedBitmap, mat);
            Core.normalize(mat, mat, 0, 255, Core.NORM_MINMAX);
            Utils.matToBitmap(mat, croppedBitmap);
            //Normalize the image
            float[] floatValues = imageProcess.preprocessImageToNormalizedFloats(croppedBitmap);
            //Classify Image to the output. 'ans' contains the resulted label number from the classifier
            Classification ans = classifier.classifyImageToOutputs(floatValues);
            //Convert the label number into respected labels
            String[] emotion = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
            //Draw Rectangle around the face with repective emotion's color
            Imgproc.rectangle(display_image, new Point((int) facesArray[i].tl().x, (int) facesArray[i].tl().y), new Point((int) facesArray[i].br().x, (int) facesArray[i].br().y), map.get(emotion[Integer.parseInt(ans.getLabel())]), 3);
            //Display the resulted labels in the image
            Imgproc.putText(display_image,emotion[Integer.parseInt(ans.getLabel())]+ans.getConf(),new Point((int) facesArray[i].tl().x, (int) facesArray[i].tl().y),Core.FONT_HERSHEY_PLAIN, 3.0, map.get(emotion[Integer.parseInt(ans.getLabel())]),3);
        }
        return display_image;
    }
    private static final int UI_ANIMATION_DELAY = 300;
    private final Handler mHideHandler = new Handler();
    private View mContentView;
    private final Runnable mHidePart2Runnable = new Runnable() {
        @SuppressLint("InlinedApi")
        @Override
        public void run() {
            // Delayed removal of status and navigation bar

            // Note that some of these constants are new as of API 16 (Jelly Bean)
            // and API 19 (KitKat). It is safe to use them, as they are inlined
            // at compile-time and do nothing on earlier devices.
            mContentView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LOW_PROFILE
                    | View.SYSTEM_UI_FLAG_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        }
    };
    private View mControlsView;

    private final Runnable mShowPart2Runnable = new Runnable() {
        @Override
        public void run() {
            // Delayed display of UI elements
            ActionBar actionBar = getSupportActionBar();
            if (actionBar != null) {
                actionBar.show();
            }
            mControlsView.setVisibility(View.VISIBLE);
        }
    };
    private boolean mVisible;
    private final Runnable mHideRunnable = new Runnable() {
        @Override
        public void run() {
            hide();
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        //Ask the permission for the camera
        //Need to procees further if the user denied to allow the access-Not done yet
        if (PermissionChecker.checkSelfPermission(this, permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
        {
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, permission.CAMERA))
            {
                //
            }
            else
            {
                ActivityCompat.requestPermissions(this,new String[]{permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA );
            }
        }
        if (OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "Open Cv Loaded Successfully", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(getApplicationContext(), "Open Cv Failed to Load", Toast.LENGTH_SHORT).show();
        }
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mVisible = true;
        mControlsView = findViewById(R.id.fullscreen_content_controls);
        mContentView = findViewById(R.id.fullscreen_content);
        // Set up the user interaction to manually show or hide the system UI.
        mContentView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                toggle();
            }
        });
        loadModel();
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        hide();

    }

    private void toggle() {
        if (mVisible) {
            hide();
        } else {
            show();
        }
    }
    private void hide() {
        // Hide UI first
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.hide();
        }
        mControlsView.setVisibility(View.GONE);
        mVisible = false;

        // Schedule a runnable to remove the status and navigation bar after a delay
        mHideHandler.removeCallbacks(mShowPart2Runnable);
        mHideHandler.postDelayed(mHidePart2Runnable, UI_ANIMATION_DELAY);
    }

    @SuppressLint("InlinedApi")
    private void show() {
        mVisible = true;
        // Schedule a runnable to display UI elements after a delay
        mHideHandler.removeCallbacks(mHidePart2Runnable);
        mHideHandler.postDelayed(mShowPart2Runnable, UI_ANIMATION_DELAY);
    }
    private void loadModel() {

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    //modelPath:- contains the .pb file built from the neural network.
                    //LabelFile:- .txt file containing the output labels of the classifier
                    //inputName :- input node in the neural network model
                    //outputName:- name of the output node in clasifier
                    //Add thesse .pb and .txt files inside the Asset Folder
                    classifier =  ImageClassifier.create(getAssets(),
                            "optimize_expression_model.pb", "labels.txt", image_size,
                            "input", "output");
                } catch (final Exception e) {
                    //if they classifier isnot found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }
    //Switch camera front and back on clicking button.
    public void switchCamera(View view){
        cameraId = cameraId^1;

        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(cameraId);
        mOpenCvCameraView.enableView();
    }
}
