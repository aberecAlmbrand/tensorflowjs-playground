import * as tf from '@tensorflow/tfjs';
import {BostonHousingDataset} from './data';
import {plotDataSimple, plotDataAndPredictions, plotDataAndPredictionsMark} from './ui';
import * as ui from './ui';
import * as normalization from './normalization';
import { MAIN } from 'vega-lite/build/src/data';


// Some hyperparameters for model training.
const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};

// Convert loaded data into tensors and creates normalized versions of the
// features.
export const arraysToTensors = () => {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
  // Normalize mean and standard deviation of data.
  let {dataMean, dataStd} =
      normalization.determineMeanAndStddev(tensors.rawTrainFeatures);

  tensors.trainFeatures = normalization.normalizeTensor(
      tensors.rawTrainFeatures, dataMean, dataStd);
  tensors.testFeatures =
      normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
};

/*
document.addEventListener('DOMContentLoaded', async () => {
    await bostonData.loadData();
    ui.updateStatus('Data loaded, converting to tensors');

    arraysToTensors();
    
    ui.updateStatus(
        'Data is now available as tensors.\n' +
        'Click a train button to begin.');


  }, false);*/

  mainLoader();
  async function mainLoader(){
    await bostonData.loadData();
    ui.updateStatus('Data loaded, converting to tensors');

    arraysToTensors();
    
    ui.updateStatus(
        'Data is now available as tensors.\n' +
        'Click a train button to begin.');


    /*const x = tf.tensor1d([1, 2, 3, 4]);
    x.slice([0]).print(); 
    x.slice([1]).print(); 
    x.slice(0).print(); 
    x.slice([1], [2]).print();   */
    
    const x2 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    x2.slice([1, 0]).print();
    x2.slice([1, 0], [1, 2]).print();

  let sliced = tensors.rawTrainFeatures.slice([0, 11]); 
  sliced.print();
   await plotDataSimple('#data .plot', sliced, tensors.trainTarget);
  let sliced2 = tensors.rawTrainFeatures.slice([1, 11]); 
  sliced2.print();
  await plotDataSimple('#data .plot2', sliced2, tensors.trainTarget);
  let sliced3 = tensors.rawTrainFeatures.slice([2, 11]); 
  sliced3.print();
  await plotDataSimple('#data .plot3', sliced3, tensors.trainTarget);

  //await plotDataSimple('#data .plot', tensors.rawTrainFeatures, tensors.trainTarget);

  }
