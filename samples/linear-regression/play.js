import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';


buildGraf();
async function buildGraf() {
    const xvalues = [];
    const yvalues = [];
    for(let i = 0; i < 10; i++){
        xvalues[i] = Math.random() * 100;
        yvalues[i] = Math.pow(xvalues[i], 2);
    }
    //row, col
    const shape = [5, 2];
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

    train(model, xs, ys).then(() => {
        let outputs = model.predict(xs);
        outputs.print();
        console.log('training complete');
    });
    
    await plotDataAndPredictions(
        '#random .plot', _xvalues, _yvalues, predictionsBefore);
}


// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model performs.
const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
    // Having a good error function is key for training a machine learning model
    const error = prediction.sub(labels).square().mean();
    return error;
  }


/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations) {
    for (let iter = 0; iter < numIterations; iter++) {
      // optimizer.minimize is where the training happens.
  
      // The function it takes must return a numerical estimate (i.e. loss)
      // of how well we are doing using the current state of
      // the variables we created at the start.
  
      // This optimizer does the 'backward' step of our training process
      // updating variables defined previously in order to minimize the
      // loss.
      optimizer.minimize(() => {
        // Feed the examples into the model
        const pred = predict(xs);
        return loss(pred, ys);
      });
  
      // Use tf.nextFrame to not block the browser.
      await tf.nextFrame();
    }
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
  for (let i = 0; i < 20; i++) {
    const config = {
      shuffle: true,
      epochs: 10
    }
    const response = await model.fit(xs, ys, config);
    console.log(response.history.loss[0]);
  }
}

