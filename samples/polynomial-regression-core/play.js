import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';


learnCoefficients();


async function learnCoefficients() {
    const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
    const trainingData = generateData(100, trueCoefficients);
  
    // Plot original data
    renderCoefficients('#data .coeff', trueCoefficients);
    await plotData('#data .plot', trainingData.xs, trainingData.ys)
}