import { lazy } from "react";

const ImitationLearningMDX = lazy(() => import("./notes/imitation-learning.mdx"));
const ValueBasedRLMDX = lazy(() => import("./notes/value-based-rl.mdx"));
const PolicyGradientsMDX = lazy(() => import("./notes/policy-gradients.mdx"));
const ActorCriticMDX = lazy(() => import("./notes/actor-critic.mdx"));
const AdvancedPolicyOptimizationMDX = lazy(() => import("./notes/advanced-policy-optimization.mdx"));
const ControlAsInferenceMDX = lazy(() => import("./notes/control-as-inference.mdx"));
const InverseRLMDX = lazy(() => import("./notes/inverse-rl.mdx"));
const ModelBasedRLMDX = lazy(() => import("./notes/model-based-rl.mdx"));
const PlanningAndMCTSMDX = lazy(() => import("./notes/planning-and-mcts.mdx"));
const OfflineRLMDX = lazy(() => import("./notes/offline-rl.mdx"));
const ExplorationAndSkillLearningMDX = lazy(() => import("./notes/exploration-and-skill-learning.mdx"));
const RLTheoryMDX = lazy(() => import("./notes/rl-theory.mdx"));

export const data = [
  {
    id: "imitation-learning",
    title: "Imitation Learning",
    component: ImitationLearningMDX,
  },
  {
    id: "value-based-reinforcement-learning",
    title: "Value-Based Reinforcement Learning",
    component: ValueBasedRLMDX,
  },
  {
    id: "policy-gradients",
    title: "Policy Gradients",
    component: PolicyGradientsMDX,
  },
  {
    id: "actor-critic-methods",
    title: "Actor-Critic Methods",
    component: ActorCriticMDX,
  },
  {
    id: "advanced-policy-optimization",
    title: "Advanced Policy Optimization",
    component: AdvancedPolicyOptimizationMDX,
  },
  {
    id: "control-as-inference",
    title: "Control as Inference and Maximum-Entropy RL",
    component: ControlAsInferenceMDX,
  },
  {
    id: "inverse-reinforcement-learning",
    title: "Reward Learning and Inverse Reinforcement Learning",
    component: InverseRLMDX,
  },
  {
    id: "model-based-reinforcement-learning",
    title: "Model-Based Reinforcement Learning",
    component: ModelBasedRLMDX,
  },
  {
    id: "planning-and-monte-carlo-tree-search",
    title: "Planning and Monte Carlo Tree Search",
    component: PlanningAndMCTSMDX,
  },
  {
    id: "offline-reinforcement-learning",
    title: "Offline Reinforcement Learning",
    component: OfflineRLMDX,
  },
  {
    id: "exploration-and-skill-learning",
    title: "Exploration and Skill Learning",
    component: ExplorationAndSkillLearningMDX,
  },
  {
    id: "reinforcement-learning-theory",
    title: "Reinforcement Learning Theory",
    component: RLTheoryMDX,
  },
];
