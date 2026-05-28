# Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment
This repository contains the official code for the ICML 2026 paper "Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment".

🚧 News: The official implementation is currently under preparation and will be open-sourced very soon! Stay tuned! 

## 📖 Abstract
Generative distillation significantly accelerates text-to-image (T2I) generation by compressing multi-step trajectories into few-step student models while preserving perceptual quality. However, existing methods primarily optimize efficiency and output fidelity, often neglecting a critical property of the original trajectory: sensitivity to initial noise. The degradation of this sensitivity impairs downstream control methods relying on noise-based optimization and manipulation. 
In this work, we propose Geometry-Aware Distillation (GAD), a sensitivity-preserving framework that aligns the local functional behavior of teacher and student models. Specifically, GAD matches Jacobian-vector products (JVP) with respect to input noise, enabling the student to reproduce the teacher's differential response to perturbations. Extensive experiments across multiple T2I paradigms demonstrate that GAD significantly restores sensitivity and improves diversity while maintaining high visual fidelity.

<p align="center">
	<img src="assets/teaser.png", width="600"

## 🌟 Key Features
- Model-Agnostic Regularizer: GAD serves as a plug-and-play regularization term that can be seamlessly integrated with existing distillation paradigms.
- Restored Noise Sensitivity: Effectively addresses the smoothing effect of standard pointwise alignment objectives, recovering the model's sensitivity to input noise.
- Enhanced Controllability: Achieves superior performance on downstream tasks dependent on initial noise manipulation, such as training-free layout control and generation diversity.

## 🚀 Getting Started (Coming Soon)
The source code, training scripts, and pre-trained checkpoints are currently being finalized for release.Prerequisites(Detailed environment setup instructions will be updated upon release)
Installation(Installation steps will be updated upon release)

## 📊 Results
By integrating GAD, distilled models naturally alleviate diversity degradation without compromising image fidelity. Our method provides a unified solution that reconciles the trade-off between inference speed and generative controllability, outperforming standard distillation baselines on noise-based control tasks.  

<p align="center">
	<img src="assets/result.png", width="800">

## ✒️ Citation
If you find our work helpful for your research, please consider citing our paper:
```
@inproceedings{huang2026restoring,
  title={Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment},
  author={Huang, Huayang and Wang, Ruoyu and Zhao, Jinhui and Deng, Wei and Zhou, Daiguo and Luan, Jian and Wu, Yu and Zhu, Ye},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026},
  organization={PMLR}
}
```
