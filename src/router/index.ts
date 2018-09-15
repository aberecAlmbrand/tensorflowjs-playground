import Vue from "vue";
import Router from "vue-router";

Vue.use(Router);

import HelloWorld from "@/components/HelloWorld.vue";

import TensorflowSample from "@/components/tensorflow";

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

  ],
});

