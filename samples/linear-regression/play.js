import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, plotDataAndPredictionsMark} from './ui';


const MODEL_SAVE_PATH_ = "localstorage://my-model-1";

let _xvalues, _yvalues;

/*********************************** */
//trainModelAndBuildLinearGraf();
/*********************************** */
async function trainModelAndBuildLinearGraf() {
    // simpel neural model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    //definer loss metode og optimizer til neural model
    model.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    });
  
    // test data med formlen y = 2x - 1
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4, 6, 7, 8, 9], [10, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7, 11, 13, 15, 17], [10, 1]);

    await plotData('#data .plot', xs, ys);

    var button = document.createElement('button');
    button.innerHTML = 'Træn neural netværk';
    button.onclick = function(){
        // træn neural model, gentag 250 gange
        model.fit(xs, ys, {epochs: 250}).then(()=>{
            let predictX = 5;
            //Få tensorflow til at forudsige y værdien udfra x på 5 => 9 = 2 * 5 - 1
            let predictedValues = model.predict(tf.tensor2d([predictX], [1, 1]));

            predictedValues.print();

            plotDataAndPredictionsMark('#trained .plot', xs, ys, predictX, predictedValues).then(()=>{
                xs.dispose();
                ys.dispose();
                predictedValues.dispose();
            });
        })
    };

    document.getElementById('foobutton').appendChild(button);

    //console.log("tensors "+tf.memory().numTensors);
  }



/*********************************** */
//trainModelAndBuildPolynomialGraf();
/*********************************** */
async function trainModelAndBuildPolynomialGraf() {

    await buildPolynomialDataSet(20);

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
    const sgdOpt = tf.train.sgd(0.1);

    //færdig med config => compile model
    model.compile({
        optimizer: sgdOpt,
        loss: tf.losses.meanSquaredError
    });

    var button = document.createElement('button');
    button.innerHTML = 'Træn neural netværk';
    button.onclick = function(){
        train(model, _xvalues, _yvalues).then(() => {
            //model er trænet => forudsig x værdier
            let predictedValues = model.predict(_xvalues);
    
            _xvalues.print();
            predictedValues.print();
            console.log('training complete');
    
            plotDataAndPredictions('#trained .plot', _xvalues, _yvalues, predictedValues).then(()=>{
                _xvalues.dispose();
                _yvalues.dispose();
                predictedValues.dispose();
            })
    
            //gem model i browser og hent det frem senere
            saveModel(model).then((result) =>{
                const saveResult = result;
                console.log(saveResult);
            });
        });
    };

    document.getElementById('foobutton').appendChild(button);
}
//træn model
async function train(model, xs, ys) {
    for (let i = 0; i < 5000; i++) {
        const config = {
            shuffle: true,
            epochs: 20
        }
        //læg data i model og vent på at Tensorflow bliver færdig med forudsigelse
        const response = await model.fit(xs, ys, config);

        if(i%10 === 0){
            let predictedValues = model.predict(xs);
            plotDataAndPredictions('#trained .plot', xs, ys, predictedValues).then(()=>{
                predictedValues.dispose();
            })
        }

        //print loss så man kan se at forudsigelsen bliver forbedret
        console.log(response.history.loss[0] + " => "+i);
        //console.log("tensors "+tf.memory().numTensors);
    }
}

async function buildPolynomialDataSet(count){
    const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
    const trainingData = generateData(count, trueCoefficients);

    let tmpXvalues = await trainingData.xs;
    let tmpYvalues = await trainingData.ys;

    _xvalues = tf.reshape(tmpXvalues, [count, 1])
    _yvalues = tf.reshape(tmpYvalues, [count, 1])

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

    const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12], [2, 12]);
    x.print();
    x.slice([0, 0], 1).print();

    const Y = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12,2, 1, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12], [2, 12]);    
    Y.print();
    Y.slice([1, 0], 1).print();
}


//loadModalAndBuildGraf();
async function loadModalAndBuildGraf(){
    
    await buildPolynomialDataSet();

    loadModel().then(model =>{
        let predictedValues = model.predict(_xvalues);

        _xvalues.print();
        predictedValues.print();
        console.log('training complete');

        plotDataAndPredictions('#trained .plot', _xvalues, _yvalues, predictedValues);
    });
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

  async function saveModel(model) {
    return await model.save(MODEL_SAVE_PATH_);
  }


  async function checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  async function removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }


/*********************************** */
//trainModelAndBuildLinearGraf2();
/*********************************** */
async function trainModelAndBuildLinearGraf2() {
    // simpel neural model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 2, inputShape: [2]}));
    
    //definer loss metode og optimizer til neural model
    model.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    });
  
    // test data med formlen y = 2x - 1
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4, 6, 7, 8, 9, -1, 0, 1, 2, 3, 4, 6, 7, 8, 9], [10, 2]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7, 11, 13, 15, 17, -6, -3, 0, 3, 6, 9, 15, 18, 21, 24], [10, 2]);

    // test data med formlen y = 3x - 3
    //const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4, 6, 7, 8, 9], [10, 1]);
    //const ys = tf.tensor2d([-6, -3, 0, 3, 6, 9, 15, 18, 21, 24], [10, 1]);

    await plotData('#data .plot', xs, ys);

    var button = document.createElement('button');
    button.innerHTML = 'Træn neural netværk';
    button.onclick = function(){
        // træn neural model, gentag 250 gange
        model.fit(xs, ys, {epochs: 250}).then(()=>{
            let predictX = 5;
            //Få tensorflow til at forudsige y værdien udfra x på 5 => 9 = 2 * 5 - 1
            let predictedValues = model.predict(tf.tensor2d([predictX, predictX], [1, 2]));

            predictedValues.print();

            plotDataAndPredictionsMark('#trained .plot', xs, ys, predictX, predictedValues).then(()=>{
                xs.dispose();
                ys.dispose();
                predictedValues.dispose();
            });
        })
    };

    document.getElementById('foobutton').appendChild(button);

    //console.log("tensors "+tf.memory().numTensors);
  }





