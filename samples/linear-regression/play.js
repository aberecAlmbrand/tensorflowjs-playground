import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';


buildGraf();
async function buildGraf() {
    const xvalues = [];
    const yvalues = [];
    for(let i = 0; i < 10; i++){
        xvalues[i] = Math.random();
        yvalues[i] = xvalues[i] + 0.1;
    }
    //row, col
    const shape = [10, 1];
    const _xvalues = tf.tensor2d(xvalues, shape);
    const _yvalues = tf.tensor2d(yvalues, shape);

    _xvalues.print();
    _yvalues.print();

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
        let outputs = model.predict(_xvalues);

        outputs.print();
        console.log('training complete');

        plotDataAndPredictions('#trained .plot', _xvalues, _yvalues, outputs);
    });

}

/******************************************************************************* */


//simpleLinearRegression();
function simpleLinearRegression(){
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    xs.print();
    ys.print();
    
    // Train the model using the data.
    model.fit(xs, ys, {epochs: 10}).then(() => {
        // Use the model to do inference on a data point the model hasn't seen before:
        (model.predict(tf.tensor2d([5], [1, 1]))).print();
    });
}






  /********************************************************************************************************** */


//simpleTensors();
function simpleTensors(){

    const values = [];
    for(let i = 0; i < 15; i++){
        values[i] = Math.random() * 100;
    }
    //row, col
    const shape = [5, 3];
    const data = tf.tensor2d(values, shape);
    data.print();

    //await plotData('#data .plot', trainingData.xs, trainingData.ys)

    const values2 = [];
    for(let i = 0; i < 30; i++){
        values2[i] = Math.random() * 100;
    }
    //matrix, row, col
    const shape2 = [2, 5, 3];
    const data2 = tf.tensor3d(values2, shape2);
    data2.print();


    tf.tensor3d([[[1], [2]], [[3], [4]]]).print();

    tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
}



  /********************************************************************************************************** */



//learnCoefficients();
async function learnCoefficients() {
    const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
    const trainingData = generateData(100, trueCoefficients);
  
    // Plot original data
    renderCoefficients('#data .coeff', trueCoefficients);
    await plotData('#data .plot', trainingData.xs, trainingData.ys);
}


  /********************************************************************************************************** */



//linearRegression();
function linearRegression(){
    // This is the model
    const model = tf.sequential();

    // Create the hidden layer
    // dense is a "full connected layer"
    const hidden = tf.layers.dense({
        units: 4, // number of nodes
        inputShape: [2], // input shape
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
    const sgdOpt = tf.train.sgd(0.1);

    // I'm done configuring the model so compile it
    model.compile({
        optimizer: sgdOpt,
        loss: tf.losses.meanSquaredError
    });


    const xs = tf.tensor2d([
        [0, 0],
        [0.5, 0.5],
        [1, 1]
    ]);

    const ys = tf.tensor2d([
        [1],
        [0.5],
        [0]
    ]);

    //await plotData('#data .plot', xs, ys)

    train(model, xs, ys).then(() => {
        let outputs = model.predict(xs);
        outputs.print();
        console.log('training complete');
    });
}

async function train(model, xs, ys) {
  for (let i = 0; i < 300; i++) {
    const config = {
      shuffle: true,
      epochs: 10
    }
    const response = await model.fit(xs, ys, config);
    console.log(response.history.loss[0]);
  }
}
