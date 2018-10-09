import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from './imagenet_classes';
import {ControllerDataset} from './controller_dataset';
import {Webcam} from './webcam';


const webcam = new Webcam(document.getElementById('webcam'));

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    //'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
    //'http://127.0.0.1:5500/tensorflowjs-playground/samples/mobilenet/model2/model.json';
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;
const COLOR = 3;
const NUM_CLASSES = 2;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

const label = "erol";

let mobilenet;
let model;

const mobilenetDemo = async () => {
  status('Loading model...');

  document.addEventListener("DOMContentLoaded", async function(){

    //mobilenet = await tf.loadModel(MOBILENET_MODEL_PATH);

    //const layer = mobilenet.getLayer('conv_pw_13_relu');
    //mobilenet = await tf.model({inputs: mobilenet.inputs, outputs: layer.output});

    await init();

   await tf.tidy(() => {
      const img = webcam.capture();
      controllerDataset.addExample(mobilenet.predict(img), label);
  
    });

    await train();

    //const layerOutput = layer.output.shape;
    //layerOutput.print();

    //1 billede af 224x224 pixels i RGB (3 channels),
    //mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, COLOR])).dispose();

    status('');

    const catElement = document.getElementById('cat');
    if (catElement.complete && catElement.naturalHeight !== 0) {
      predict2(catElement);
      catElement.style.display = '';
    } else {
      catElement.onload = () => {
        predict2(catElement);
        catElement.style.display = '';
      }
    }

    document.getElementById('file-container').style.display = '';

  });
};



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
  /*const batchSize = Math.floor(controllerDataset.xs.shape[0] * 0.4);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }*/
  let batchSize = 10;

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: 20,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');

  const startTime = performance.now();
  const logits = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.fromPixels(imgElement).toFloat();

    //img.print();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    //normalized.print();

    // Reshape to a single-element batch so we can pass it to predict.
    //1 billede af 224x224 pixels i RGB (3 channels),
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, COLOR]);

    //batched.print();

    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime = performance.now() - startTime;
  status(`Done in ${Math.floor(totalTime)}ms`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

async function predict2(img) {
  const predictedClass = tf.tidy(() => {
    // Capture the frame from the webcam.
    const _img = webcam.uploadImage(img);

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model.
    const activation = mobilenet.predict(_img);

    // Make a prediction through our newly-trained model using the activation
    // from mobilenet as input.
    const predictions = model.predict(activation);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    return predictions.as1D().argMax();
  });
}




/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
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
      img.onload = () => predict2(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
