package com.example.kiran.FacialExpression;
import android.os.Trace;
import android.content.res.AssetManager;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
public class ImageClassifier {
    private List<String> labels;
    private String[] outputNames;
    private static final float THRESHOLD = 0.1f;
    private String inputName;
    private String outputName;
    private float[] results;
    private TensorFlowInferenceInterface tf;
    private int inputSize;

    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(am.open(fileName)));
        String line;
        List<String> labels = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }
        br.close();
        return labels;
    }

    public static ImageClassifier create(AssetManager assetManager,
                                         String modelPath, String labelFile, int inputSize, String inputName, String outputName) throws IOException {
        //intialize a classifier
        ImageClassifier c = new ImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;
        //read labels for label file
        c.labels = readLabels(assetManager, labelFile);
        //set its model path and where the raw asset files are
        c.tf = new TensorFlowInferenceInterface(assetManager, modelPath);
        int numClasses = 7; //This is the number of output classes in the model. We are classifying into 7 classes.
        c.inputSize = inputSize;
        // Pre-allocate buffer.
        c.outputNames = new String[]{outputName};
        c.outputName = outputName;
        c.results = new float[numClasses];
        return c;
    }

    public Classification classifyImageToOutputs(final float[] Pixels) {
        //Some tf function may be depreciated and may not support in some cases. In such cases use the alternatives now commented just below.
        Trace.beginSection("fillNodefloat");
        tf.feed(inputName, Pixels, 1, inputSize, inputSize, 1);
        // tf.fillNodeFloat(inputName, new int[]{imageSize * imageSize}, imageNormalizedPixels);
        Trace.endSection();
        Trace.beginSection("runInference");
        tf.run(outputNames);
        //tf.runInference(new String[] {outputName});
        Trace.endSection();
        Trace.beginSection("readNodeFloat");
        //tf.readNodeFloat(outputName,results);
        tf.fetch(outputName, results);
        Trace.endSection();
        Classification ans = new Classification();
        //Get the label with highest conf.
        for (int i = 0; i < results.length; i++) {
            if (results[i] > THRESHOLD && results[i] > ans.getConf()) {
                ans.update(results[i], labels.get(i));
            }
        }
        return ans;
    }
}