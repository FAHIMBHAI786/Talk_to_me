package com.example.talktome;

import android.content.Intent;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;


import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.talktome.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements TextToSpeech.OnInitListener {

    private static final int requestCode =1;
    ImageButton cameraButton;
    ImageView capturedImg;

    EditText messageBox;

    ImageButton sendButton;

    private TextToSpeech textToSpeech;

    int imageSize = 32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        cameraButton = findViewById(R.id.cameraButton);
        capturedImg = findViewById(R.id.capturedImg);
        messageBox = findViewById(R.id.messageBox);
        sendButton = findViewById(R.id.sendButton);

        cameraButton.setOnClickListener(v -> {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityIfNeeded(cameraIntent, requestCode);
        });

        textToSpeech = new TextToSpeech(this, this);

        sendButton.setOnClickListener(v -> {

            String text = messageBox.getText().toString();
            speak(text);
        });


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == MainActivity.requestCode && resultCode == RESULT_OK)
        {
            assert data != null;
            Bitmap image = (Bitmap) data.getExtras().get("data") ;
            assert image != null;
            int dimension = Math.min(image.getWidth(),image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
            
//            assert data != null;
//            Bundle extras = data.getExtras();
//            assert extras != null;
            capturedImg.setImageBitmap(image);
            image = Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int [] intValues = new int[imageSize * imageSize];

            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
           int pixel = 0;
            for (int i = 0; i<imageSize; i++){
                for (int j = 0; j< imageSize; j++){

                    int val = intValues[pixel++];//RGB
                    byteBuffer.putFloat(((val >> 16)& 0xFF)*(1.f/1));
                    byteBuffer.putFloat(((val >> 8)& 0xFF)*(1.f/1));
                    byteBuffer.putFloat((val & 0xFF)*(1.f/1));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float [] confidence = outputFeature0.getFloatArray();
            int maxIndex = 0;
            float maxConfidence = 0;

            for (int i = 0; i < confidence.length; i++) {
                if (confidence[i] > maxConfidence) {
                    maxConfidence = confidence[i];
                    maxIndex = i;
                }
            }
            String[] classes = {"Apple", "Banana", "Orange"};
            String result = classes[maxIndex];
            if (result.equals("Apple") || result.equals("Banana") || result.equals("Orange")){
            speak("I Think This is an image of " + result);}

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            int result = textToSpeech.setLanguage(Locale.UK);
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "Language not supported");
            } else {
                // TTS is ready to use
                speak("Hello, welcome to talk to me a i !");
            }
        } else {
            Log.e("TTS", "Initialization failed");
        }
    }

    private void speak(String text) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
    }

    @Override
    protected void onDestroy() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onDestroy();
    }
}