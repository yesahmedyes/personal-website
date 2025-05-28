/* eslint-disable @typescript-eslint/no-unsafe-return */
import { lazy } from "react";

const LinearRegressionMDX = lazy(() => import("./notes/linear-regression.mdx"));
const ClassificationMDX = lazy(() => import("./notes/classification.mdx"));
const GenerativeLearningAlgorithmsMDX = lazy(() => import("./notes/gla.mdx"));
const KernelMethodsMDX = lazy(() => import("./notes/kernel-methods.mdx"));
const LearningTheoryMDX = lazy(() => import("./notes/learning-theory.mdx"));
const ClusteringMDX = lazy(() => import("./notes/clustering.mdx"));
const PrincipalComponentAnalysisMDX = lazy(() => import("./notes/pca.mdx"));
const IndependentComponentAnalysisMDX = lazy(() => import("./notes/ica.mdx"));
const EMAlgorithmMDX = lazy(() => import("./notes/em-algorithm.mdx"));
const GaussianMixtureModelsMDX = lazy(() => import("./notes/gmm.mdx"));
const FactorAnalysisMDX = lazy(() => import("./notes/factor-analysis.mdx"));
const VariationalAutoencodersMDX = lazy(() => import("./notes/vae.mdx"));
const DecisionTreesMDX = lazy(() => import("./notes/decision-trees.mdx"));
const ReinforcementLearningMDX = lazy(() => import("./notes/rl.mdx"));

export const data = [
  {
    id: "linear-regression",
    title: "Linear Regression",
    component: LinearRegressionMDX,
  },
  {
    id: "classification",
    title: "Classification",
    component: ClassificationMDX,
  },
  {
    id: "generative-learning-algorithms",
    title: "Generative Learning Algorithms",
    component: GenerativeLearningAlgorithmsMDX,
  },
  {
    id: "kernel-methods",
    title: "Kernel Methods",
    component: KernelMethodsMDX,
  },
  {
    id: "learning-theory",
    title: "Learning Theory",
    component: LearningTheoryMDX,
  },
  {
    id: "clustering",
    title: "Clustering",
    component: ClusteringMDX,
  },
  {
    id: "principal-component-analysis",
    title: "Principal Component Analysis",
    component: PrincipalComponentAnalysisMDX,
  },
  {
    id: "independent-component-analysis",
    title: "Independent Component Analysis",
    component: IndependentComponentAnalysisMDX,
  },
  {
    id: "em-algorithm",
    title: "Expectancy Maximization Algorithm",
    component: EMAlgorithmMDX,
  },
  {
    id: "gaussian-mixture-models",
    title: "Gaussian Mixture Models",
    component: GaussianMixtureModelsMDX,
  },
  {
    id: "factor-analysis",
    title: "Factor Analysis",
    component: FactorAnalysisMDX,
  },
  {
    id: "variational-autoencoders",
    title: "Variational Autoencoders",
    component: VariationalAutoencodersMDX,
  },
  {
    id: "decision-trees",
    title: "Decision Trees",
    component: DecisionTreesMDX,
  },
  {
    id: "reinforcement-learning",
    title: "Reinforcement Learning",
    component: ReinforcementLearningMDX,
  },
];
