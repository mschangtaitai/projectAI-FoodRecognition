package com.chickensaladItalianVersion

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.chickensaladItalianVersion.ml.ItalianModelNew
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {

    //lateinit var selecthola: Button
    lateinit var textOutput : TextView
    lateinit var bitmap: Bitmap
    //a
    lateinit var albumButton: Button
    lateinit var predictButton: Button
    lateinit var imgView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        albumButton = findViewById(R.id.button_album)
        //Log.d
        predictButton = findViewById(R.id.button2)
        //predictButton = findViewById(R.id.button2)
        imgView = findViewById(R.id.imageView)
        textOutput = findViewById(R.id.textView)
        val labels = application.assets.open("dict.txt").bufferedReader().use { it.readText() }.split("\n")
        print(labels)

        albumButton.setOnClickListener(View.OnClickListener {
            Log.d("mssg", "button pressed")
            var intent : Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 100)
        })

        predictButton.setOnClickListener(View.OnClickListener {
            val tfImage = Bitmap.createScaledBitmap(bitmap, 180, 180, true)

            val byteBuffer: ByteBuffer = ByteBuffer.allocate(180*180*3*4)
            byteBuffer.rewind()
            if (tfImage != null) {
                tfImage.copyPixelsToBuffer(byteBuffer)
            }

            val model = ItalianModelNew.newInstance(this)
            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 180, 180, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)
            val outputs = model.process(inputFeature0)

            val outputFeature0 = outputs.outputFeature0AsTensorBuffer



            val scores = outputFeature0.floatArray
            val prediction = prediction(scores)
            Toast.makeText(this,"output: "+prediction,Toast.LENGTH_SHORT).show()
// Releases model resources if no longer used.
            model.close()
        })

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        imgView.setImageURI(data?.data)
        var uri : Uri ?= data?.data
        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

    }




    fun prediction(scores: FloatArray):String{
     //   var pred1 = scores.base
        var pred = ""
        var max = 0.0f
        //Log.d("msg", imgView.toString())
        Log.d("warning", scores.size.toString())
        Log.d("warning", scores.get(1).toString())
        Log.d("warning", scores.get(2).toString())
        Log.d("warning", scores.get(3).toString())
        Log.d("warning", scores.get(4).toString())
        if (scores.max() == scores.get(0)) {
            return "lasagna"
        }
        if (scores.max() == scores.get(1) ) {
            return "cannoli"
        }
        if (scores.max() == scores.get(2)) {
            return "tiramisu"
        }
        if (scores.max() == scores.get(3)) {
            return "rissoto"
        }
        if (scores.max() == scores.get(4)) {
            return "pizza"
        } else {
            return "con ninguno"
        }
    }
}





































