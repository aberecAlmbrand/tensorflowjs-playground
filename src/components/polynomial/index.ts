import Vue from "vue";
import { Component, Emit, Inject, Model, Prop, Provide, Watch } from "vue-property-decorator";


import template from './Polynomial.vue';
import * as tf from '@tensorflow/tfjs';
//import renderChart from 'vega-embed';
import MyData from './MyData';
import {generateData} from '../providers/data';

interface trainingDataType{
    xs:any,
    ys:any
}

@Component({
    name: 'Polynomial-component',
    mixins: [template],
    components: {}
})
export default class Polynomial extends Vue {

    trainingData: trainingDataType = {
        xs : null,
        ys : null
    }


    mounted() {



    }

    async handleData(){
        const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
        const trainingData = generateData(100, trueCoefficients);

       // const xvals = await trainingData.xs.data();
       // const yvals = await trainingData.ys.data();
    }
 
   async plotData(container: any, xs:any, ys:any) {
        const xvals = await xs.data();
        const yvals = await ys.data();

        /*let data: MyData[] = [];
        const values = Array.from(yvals).map((y, i) => {
            data.push(new MyData(xvals[i], yvals[i]));
        });*/
        
        const values = Array.from(yvals).map((y, i) => {
            return {'x': xvals[i], 'y': yvals[i]};
        });
     
        const spec = {
          '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
          'width': 300,
          'height': 300,
          //'data': {'values': values},
          'mark': 'point',
          'encoding': {
            'x': {'field': 'x', 'type': 'quantitative'},
            'y': {'field': 'y', 'type': 'quantitative'}
          }
        };


     
        //return renderChart(container, spec, {actions: false});
      }

  

}