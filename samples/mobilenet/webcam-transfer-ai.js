import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from './imagenet_classes';
import {ControllerDataset} from './controller_dataset';
import {Webcam} from './webcam';


const webcam = new Webcam(document.getElementById('webcam'));

const MOBILENET_MODEL_PATH =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const NUM_CLASSES = 3;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;


async function traning(label){
  await tf.tidy(() => {
    console.log("started => "+label);
    status("started => "+label);
    for(let i = 0; i < 100; i ++){
        //setTimeout( function timer(){
         const img = webcam.capture();
         controllerDataset.addExample(mobilenet.predict(img), label);

         // Draw the preview thumbnail.
         drawThumb(img, label);

      //}, i*200 );
    }
  });

  console.log("done => "+label);
  status("done => "+label);
}

const mobilenetDemo = async () => {
  status('Loading model...');

  document.addEventListener("DOMContentLoaded", async function(){

    await init();
    
    let button = document.createElement('button');
    button.innerHTML = 'Tilføj billeder 0';
    button.onclick = function(){
      traning(0);
    }
    document.getElementById('addImages').appendChild(button);

    button = document.createElement('button');
    button.innerHTML = 'Tilføj billeder 1';
    button.onclick = function(){
      traning(1);
    }
    document.getElementById('addImages2').appendChild(button);

    button = document.createElement('button');
    button.innerHTML = 'Tilføj billeder 2';
    button.onclick = function(){
      traning(2);
    }
    document.getElementById('addImages3').appendChild(button);

    let button2 = document.createElement('button');
    button2.innerHTML = 'Træn neural netværk';
    button2.onclick = function(){
       train();
    }
    document.getElementById('train').appendChild(button2);

    document.getElementById('file-container').style.display = '';

  });
};


function drawThumb(img, label) {
  /*if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    draw(img, thumbCanvas);
  }*/

  const thumbCanvas = document.getElementById('my-thumb-'+label);
  draw(img, thumbCanvas);
}

function draw(image, canvas) {
  const [width, height] = [IMAGE_SIZE, IMAGE_SIZE];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}


async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    document.getElementById('no-webcam').style.display = 'block';
  }
  
  mobilenet = await loadMobilenet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(MOBILENET_MODEL_PATH);

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}


/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      //tf.layers.flatten({inputShape: [224,224,3]}),
      // Layer 1
      tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.0001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize = Math.floor(controllerDataset.xs.shape[0] * 0.4);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: 20,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('Loss: ' + logs.loss.toFixed(5));
        status('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

async function predict(img) {
  const predictedClass = tf.tidy(() => {
    // Capture the frame from the webcam.
    const _img = webcam.uploadImage(img);

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model.
    const activation = mobilenet.predict(_img);
    //activation.print();

    // Make a prediction through our newly-trained model using the activation
    // from mobilenet as input.
    const predictions = model.predict(activation);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    return predictions.as1D().argMax();
  });
  //console.log("predictedClass: "+predictedClass);
  predictedClass.print();

 document.getElementById('result').innerHTML = "Billede: "+await predictedClass.data() +" fundet";

}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
