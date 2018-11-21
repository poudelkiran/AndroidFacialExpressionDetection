package com.example.kiran.FacialExpression;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;

public class ImageProcess {
    public static final Bitmap CropBitmap(Bitmap bitmap,int size) {
        //Crop the bitmap to the imageSize (48 in our case)
        int image_height = bitmap.getHeight();
        int image_width = bitmap.getWidth();
        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Matrix frameToCropTransformations = getTransformationMatrix(image_width, image_height, size, size, 0, false);
        Matrix cropToFrameTransformations = new Matrix();
        frameToCropTransformations.invert(cropToFrameTransformations);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(bitmap, frameToCropTransformations, null);
        return croppedBitmap;
    }
    private static final Matrix getTransformationMatrix(int srcWidth, int srcHeight, int dstWidth, int dstHeight, int applyRotation, boolean maintainAspectRatio) {
            Matrix matrix = new Matrix();
            matrix.postTranslate((float)(-srcWidth) / 2.0F, (float)(-srcHeight) / 2.0F);
            matrix.postRotate((float)applyRotation);
            boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
            int inWidth = transpose ? srcHeight : srcWidth;
            int inHeight = transpose ? srcWidth : srcHeight;
            if (inWidth != dstWidth || inHeight != dstHeight) {
                float scaleFactorX = (float)dstWidth / (float)inWidth;
                float scaleFactorY = (float)dstHeight / (float)inHeight;
                if (maintainAspectRatio) {
                    float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                    matrix.postScale(scaleFactor, scaleFactor);
                } else {
                    matrix.postScale(scaleFactorX, scaleFactorY);
                }
            }
            matrix.postTranslate((float)dstWidth / 2.0F, (float)dstHeight / 2.0F);
            return matrix;
        }

    public static float[] preprocessImageToNormalizedFloats(Bitmap bitmap) {
        //Convert pixels to float after normalizing.
        int size = 48;
        float mean = 127.5f;
        float std = 1.0f;
        int n = bitmap.getWidth()*bitmap.getHeight();
//        float[] output = new float[n*3];
        int[] intValues = new int[n];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        float[] fArray= new float[n];


        //Follow below commented code, If we need to process color image. Since we are using grayscale image, we need to process only one channel.
        for(int y=0; y<n;y++){
            fArray[y]=intValues[y]/255;
        }
//        for (int i = 0; i < intValues.length; ++i) {
//            final int val = intValues[i];
//
//            output[i] = (((val >> 16) & 0xFF) - mean)/std;
//            output[i * 3 + 1] = (((val >> 8) & 0xFF) - mean)/std;
//            output[i * 3 + 2] = ((val & 0xFF) - mean)/std;
//        }

        return fArray;

    }

}

