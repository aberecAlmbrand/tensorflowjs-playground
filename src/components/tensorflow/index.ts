import Vue from "vue";
import { Component, Emit, Inject, Model, Prop, Provide, Watch } from "vue-property-decorator";


import template from './TensorflowSample.vue';
import * as tf from '@tensorflow/tfjs';



@Component({
    name: 'TensorflowSample-component',
    mixins: [template],
    components: {}
})
export default class TensorflowSample extends Vue {

    
    mounted() {
        // Predict output for input of 2
        const result = this.predict(2);
        result.print() // Output: 24
    }
 
    linearRegression(){
        // Define a model for linear regression.
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 1, inputShape: [1]}));
        
        // Prepare the model for training: Specify the loss and the optimizer.
        model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
        
        // Generate some synthetic data for training.
        const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
        const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
        
        // Train the model using the data.
        model.fit(xs, ys, {epochs: 10}).then(() => {
            // Use the model to do inference on a data point the model hasn't seen before:
            (model.predict(tf.tensor2d([5], [1, 1]))as tf.Tensor).print();
        });
    }

    // Define function
    predict(input:number) {

        // Define constants: y = 2x^2 + 4x + 8
        const a = tf.scalar(2);
        const b = tf.scalar(4);
        const c = tf.scalar(8);

        // y = a * x ^ 2 + b * x + c
        // More on tf.tidy in the next section
        return tf.tidy(() => {
            const x = tf.scalar(input);
        
            const ax2 = a.mul(x.square());
            const bx = b.mul(x);
            const y = ax2.add(bx).add(c);
        
            return y;
        })

    }
  

}