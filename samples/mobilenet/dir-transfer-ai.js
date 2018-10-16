import * as tf from '@tensorflow/tfjs';

import {ControllerDataset} from './controller_dataset';
import {Webcam} from './webcam';
import { isNullOrUndefined } from 'util';


const webcam = new Webcam(document.getElementById('webcam'));

const MOBILENET_MODEL_PATH ='https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
//const MOBILENET_MODEL_PATH ='http://127.0.0.1:5500/tensorflowjs-playground/samples/mobilenet/data/mobilenet_v1_0.25_224.json';

const IMAGES_PATH = "http://127.0.0.1:5500/tensorflowjs-playground/samples/mobilenet/data/koerekort/";    
const IMAGES_PATH2 = "http://127.0.0.1:5500/tensorflowjs-playground/samples/mobilenet/data/cars/";   
const IMAGES_PATH3 = "http://127.0.0.1:5500/tensorflowjs-playground/samples/mobilenet/data/house/";   

const LOCAL_MODEL_SAVE_PATH_ = "http://127.0.0.1:5500/tensorflowjs-playground/samples/mobilenet/data/model/dir-transfer-ai-model-1.json";
const MODEL_SAVE_PATH_ = "indexeddb://dir-transfer-ai-model-1";
//const MODEL_SAVE_PATH_ = "downloads://dir-transfer-ai-model-1";

const IMAGE_SIZE = 224;
const NUM_CLASSES = 3;
const DIR_SIZE = [36, 12, 18];
const LABELS = ["0", "1", "2"];
const DESC_WRAPPER = ["KØREKORT", "BIL", "HUS"];

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;

let storedModelStatusInput = document.getElementById('stored-model-status');
let deleteStoredModelButton = document.getElementById('delete-stored-model');
let learnStoredModelButton = document.getElementById('learn-stored-model');
let loadLocalModelButton = document.getElementById('load-local-model');


async function loadImage(imageUrl) {
  const img = new Image();
  const promise = new Promise((resolve, reject) => {
    img.crossOrigin = '';
    img.width = IMAGE_SIZE;
    img.height = IMAGE_SIZE;
    img.onload = () => {
      resolve(img);
    };
  });

  img.src = imageUrl;
  return promise;
}

const mobilenetDemo = async () => {
  status('Loading model...');

    deleteStoredModelButton.addEventListener('click', async () => {
      if (confirm(`Are you sure you want to delete the locally-stored model?`)) {
          await removeModel();

          storedModelStatusInput.value = 'No stored model.';
          deleteStoredModelButton.disabled = true;
          learnStoredModelButton.disabled = false;
      }
    });

    learnStoredModelButton.addEventListener('click', async () => {
      if (confirm(`Are you sure you want to learn to the locally-stored model?`)) {
          learn();
      }
    });

    loadLocalModelButton.addEventListener('click', async () => {
      if (confirm(`Are you sure you want to load the locally-stored model?`)) {
        model = await loadLocalModel();
      }
    });


    await init();

    let modelStatus = await checkStoredModelStatus();
    if (!isNullOrUndefined(modelStatus)) {
      status('Loaded network from IndexedDB.');

      storedModelStatusInput.value = `Saved@${modelStatus.dateSaved.toISOString()}`;
      deleteStoredModelButton.disabled = false;
      learnStoredModelButton.disabled = true;

      model = await loadModel();
    }
};

async function learn(){

  storedModelStatusInput.value = 'No stored model.';
  deleteStoredModelButton.disabled = true;
  learnStoredModelButton.disabled = false;

    for(let i=1; i<=DIR_SIZE[0]; i++){
      let image = await loadImage(IMAGES_PATH+"images ("+i+").jpg");
      const img = webcam.uploadImage(image);
      controllerDataset.addExample(mobilenet.predict(img), LABELS[0]);
      drawThumb(img, LABELS[0]);
    }

    for(let i=1; i<=DIR_SIZE[1]; i++){
      let image = await loadImage(IMAGES_PATH2+"cars ("+i+").jpg");
      const img = webcam.uploadImage(image);
      controllerDataset.addExample(mobilenet.predict(img), LABELS[1]);
      drawThumb(img, LABELS[1]);
    }

    for(let i=1; i<=DIR_SIZE[2]; i++){
      let image = await loadImage(IMAGES_PATH3+"house ("+i+").jpg");
      const img = webcam.uploadImage(image);
      controllerDataset.addExample(mobilenet.predict(img), LABELS[2]);
      drawThumb(img, LABELS[2]);
    }

    //træn netværk
    await train();
}

function drawThumb(img, label) {
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
        status('Træner netværk => Loss: ' + logs.loss.toFixed(5));
      },
      onTrainEnd: async () =>{
          //gem model
          saveModel().then(async(result) =>{
            console.log(result);
            let modelStatus = await checkStoredModelStatus();
            if (modelStatus != null) {
              status('Loaded network from IndexedDB.');
              storedModelStatusInput.value = `Saved@${modelStatus.dateSaved.toISOString()}`;
              deleteStoredModelButton.disabled = false;
              learnStoredModelButton.disabled = true;
            }
          });
      }
    }
  });
}

async function predict(img) {
  const predictedClass = tf.tidy(() => {
    document.getElementById('error').innerHTML = "";

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
    
    //return predictions.as1D().argMax();
    return predictions.as1D();
  });
  //console.log("predictedClass: "+predictedClass);
  predictedClass.print();

  let max = predictedClass.max().dataSync();
 document.getElementById('result').innerHTML = 
 "Bedst match billede: "+ DESC_WRAPPER[predictedClass.argMax().dataSync()] +" match procent: "+max;

 if(max <= 0.99){
  document.getElementById('error').innerHTML = "Match procent ikke høj nok!!!"
 }
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

async function loadModel() {
  const modelsInfo = await tf.io.listModels();
  if (MODEL_SAVE_PATH_ in modelsInfo) {
    console.log(`Loading existing model...`);
    const model = await tf.loadModel(MODEL_SAVE_PATH_);
    console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
    return model;
  } else {
    //throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
  }
}

async function loadLocalModel() {
    console.log(`Loading existing model...`);
    const model = await tf.loadModel(LOCAL_MODEL_SAVE_PATH_);
    console.log(`Loaded model from ${LOCAL_MODEL_SAVE_PATH_}`);
    storedModelStatusInput.value = ''+LOCAL_MODEL_SAVE_PATH_;
    return model;
}

async function saveModel() {
  return await model.save(MODEL_SAVE_PATH_);
}

async function checkStoredModelStatus() {
  const modelsInfo = await tf.io.listModels();
  return modelsInfo[MODEL_SAVE_PATH_];
}

async function removeModel() {
  if(await checkStoredModelStatus() === null){
    return;
  }
  return await tf.io.removeModel(MODEL_SAVE_PATH_);
}

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

mobilenetDemo();
