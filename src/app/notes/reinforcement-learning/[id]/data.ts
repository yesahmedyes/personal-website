/* eslint-disable @typescript-eslint/no-unsafe-return */
import { lazy } from "react";

const MDPMDX = lazy(() => import("./notes/mdp.mdx"));

export const data = [
  {
    id: "mdp",
    title: "Markov Decision Processes",
    component: MDPMDX,
  },
];
