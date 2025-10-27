# 🧠 BraiNav: Incorporating Human Brain Activity to Enhance Robustness in Embodied Visual Navigation

Official implementation of the paper:

> **BraiNav: Incorporating Human Brain Activity to Enhance Robustness in Embodied Visual Navigation**  
> *Science China Technological Sciences, 2025*  
> [Daniel Pang](https://github.com/danelpeng) et al.

---

## 🧩 0. Method Overview

BraiNav introduces a **brain-informed embodied navigation framework**, integrating human EEG signals into the navigation policy to improve **robustness under visual perturbations**.  
The system aligns visual and neural representations through a multimodal encoder and trains a navigation policy that generalizes to unseen scenes and degraded visual inputs.

### 🧠 Framework Overview
<p align="center">
  <img src="assets/BraiNav_overall.pdf" alt="BraiNav Framework" width="700">
</p>


### 📈 Main Results


> 📊 BraiNav consistently improves navigation robustness under visual degradation scenarios.

---

## ⚙️ 1. Environment Setup
We recommend following the setup style of [ROBUSTNAV](https://github.com/allenai/robustnav).


## 🚀 2. Training and Evaluation

🔧 Training
bash train_navigation_agents.sh

🧪 Evaluation
bash eval_navigation_agents.sh

## 📚 3. Citation


## 🙏 Acknowledgements