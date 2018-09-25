import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';


const MODEL_SAVE_PATH_ = "localstorage://my-model-1";

const xvalues = [];
const yvalues = [];
for (let i = 0; i < 10; i++) {
    xvalues[i] = Math.random();
    yvalues[i] = xvalues[i] + 0.1;
}
//row, col
const shape = [10, 1];
let _xvalues = tf.tensor2d(xvalues, shape);
let _yvalues = tf.tensor2d(yvalues, shape);

//let _xvalues, _yvalues;

buildPolynomialModel();
async function buildPolynomialModel(){
    const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
    const trainingData = generateData(10, trueCoefficients);

    let tmpXvalues = await trainingData.xs;
    let tmpYvalues = await trainingData.ys;

    for (let i = 0; i < 10; i++) {
        xvalues[i] = tmpXvalues[i];
    }

    for (let i = 0; i < 10; i++) {
        yvalues[i] = tmpYvalues[i];
    }

    let _xvalues = tf.tensor2d(xvalues, shape);
    let _yvalues = tf.tensor2d(yvalues, shape);

    _xvalues.print();
    _yvalues.print();
    console.log('values before prediction');
}


//simpleTensors();
function simpleTensors() {

    const values = [];
    for (let i = 0; i < 15; i++) {
        values[i] = Math.random() * 100;
    }
    //row, col
    const shape = [5, 3];
    const data = tf.tensor2d(values, shape);
    data.print();

    const values2 = [];
    for (let i = 0; i < 30; i++) {
        values2[i] = Math.random() * 100;
    }
    //matrix, row, col
    const shape2 = [2, 5, 3];
    const data2 = tf.tensor3d(values2, shape2);
    data2.print();


    tf.tensor3d([[[1], [2]], [[3], [4]]]).print();

    tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
}

//myFirstTfjs();
async function myFirstTfjs() {
    // simpel neural model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    //definer loss metode og optimizer til neural model
    model.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    });
  
    // test data med formlen y = 2x - 1
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
  
    // træn neural model, gentag 250 gange
    await model.fit(xs, ys, {epochs: 250});
  
    //Få tensorflow til at forudsige y værdien udfra x på 20 => 39 = 2 * 20 - 1
    model.predict(tf.tensor2d([20], [1, 1])).print();
  }


//trainModelAndBuildGraf();
async function trainModelAndBuildGraf() {

    //buildPolynomialModel();

    //removeModel();

    await plotData('#data .plot', _xvalues, _yvalues);

    //Neural model
    const model = tf.sequential();

    // Opret hidden lag
    // dense: alle noder er forbundet med hinanden
    const hidden = tf.layers.dense({
        units: 4, // node antal
        inputShape: [1], // input shape
        activation: 'sigmoid'
    });
    // tilføj til model
    model.add(hidden);

    // ny lag
    const output = tf.layers.dense({
        units: 1,
        // input kommer fra sidste lag
        //inputShape
        activation: 'sigmoid'
    });
    model.add(output);

    // Optimizer med gradient descent
    const sgdOpt = tf.train.sgd(0.5);

    // I'm done configuring the model so compile it
    model.compile({
        optimizer: sgdOpt,
        loss: tf.losses.meanSquaredError
    });

    train(model, _xvalues, _yvalues).then(() => {
        //model er trænet => forudsig x værdier
        let predictedValues = model.predict(_xvalues);

        _xvalues.print();
        predictedValues.print();
        console.log('training complete');

        plotDataAndPredictions('#trained .plot', _xvalues, _yvalues, predictedValues);

        //gem model i browser og hent det frem senere
        saveModel(model).then((result) =>{
            const saveResult = result;
            console.log(saveResult);
        });
    });

}
//træn model
async function train(model, xs, ys) {
    for (let i = 0; i < 10; i++) {
        const config = {
            shuffle: true,
            epochs: 10
        }
        //læg data i model og vent på at Tensorflow bliver færdig med forudsigelse
        const response = await model.fit(xs, ys, config);
        //print loss så man kan se at forudsigelsen bliver forbedret
        console.log(response.history.loss[0] + " => "+i);
    }
}



//loadModalAndBuildGraf();
async function loadModalAndBuildGraf(){
    loadModel().then(model =>{
        let predictedValues = model.predict(_xvalues);

        _xvalues.print();
        predictedValues.print();
        console.log('training complete');

        plotDataAndPredictions('#trained .plot', _xvalues, _yvalues, predictedValues);
    });
}


  async function saveModel(model) {
    return await model.save(MODEL_SAVE_PATH_);
  }


  async function loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`);
      const model = await tf.loadModel(MODEL_SAVE_PATH_);
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
      return model;
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
    }
  }


  async function checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }


  async function removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }





