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
const _xvalues = tf.tensor2d(xvalues, shape);
const _yvalues = tf.tensor2d(yvalues, shape);

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

trainModelAndBuildGraf();
async function trainModelAndBuildGraf() {

    removeModel();

    _xvalues.print();
    _yvalues.print();
    console.log('values before prediction');

    await plotData('#data .plot', _xvalues, _yvalues);

    // This is the model
    const model = tf.sequential();

    // Create the hidden layer
    // dense is a "full connected layer"
    const hidden = tf.layers.dense({
        units: 4, // number of nodes
        inputShape: [1], // input shape
        activation: 'sigmoid'
    });
    // Add the layer
    model.add(hidden);

    // Create another layer
    const output = tf.layers.dense({
        units: 1,
        // here the input shape is "inferred from the previous layer"
        activation: 'sigmoid'
    });
    model.add(output);

    // An optimizer using gradient descent
    const sgdOpt = tf.train.sgd(0.5);

    // I'm done configuring the model so compile it
    model.compile({
        optimizer: sgdOpt,
        loss: tf.losses.meanSquaredError
    });

    train(model, _xvalues, _yvalues).then(() => {
        //model is trained, predict test data
        let predictedValues = model.predict(_xvalues);

        _xvalues.print();
        predictedValues.print();
        console.log('training complete');

        plotDataAndPredictions('#trained .plot', _xvalues, _yvalues, predictedValues);

        //save trained model to browser
        saveModel(model).then((result) =>{
            const saveResult = result;
            console.log(saveResult);
        });
    });

}

async function train(model, xs, ys) {
    for (let i = 0; i < 100; i++) {
        const config = {
            shuffle: true,
            epochs: 10
        }
        const response = await model.fit(xs, ys, config);
        console.log(response.history.loss[0]);
    }
}

  /**
   * Save the model to IndexedDB.
   */
  async function saveModel(model) {
    return await model.save(MODEL_SAVE_PATH_);
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns {SaveablePolicyNetwork} The instance of loaded
   *   `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
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

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  async function checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async function removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }


/*
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
*/


