import Vue from "vue";
import Router from "vue-router";

Vue.use(Router);

import HelloWorld from "@/components/HelloWorld.vue";

import TensorflowSample from "@/components/tensorflow";
import Polynomial from "@/components/polynomial";

export default new Router({
  routes: [
    {
      path: "/",
      name: "HelloWorld",
      component: HelloWorld,
    },
    {
      path: "/tensorflow-sample",
      name: "TensorflowSample",
      component: TensorflowSample,
    },
    {
      path: "/polynomial",
      name: "Polynomial",
      component: Polynomial,
    },
  ],
});

