# Audio-Visual Cross-Modal Retrieval in Remote Sensing

A B.Tech project by **Manav Jobanputra (23UCS638)** and **Pranav Khunt (23UCS671)**  
Under the guidance of **Dr. Ankit Jha**  
Department of Computer Science and Engineering, LNMIIT Jaipur — April 2026

---

## Overview

This project investigates **cross-modal retrieval between natural ambient audio and aerial imagery** using the [ADVANCE dataset](https://zenodo.org/record/3828124). Given an audio clip of environmental sounds (waves, birds, traffic), the goal is to retrieve the most relevant aerial image — and vice versa.

The core contribution is the adaptation of the **SARCI** (Scale-aware Adaptive Refinement and Cross-Interaction) framework for natural audio by replacing its speech-oriented TDNN encoder with **CNN14** pretrained on AudioSet, which is better suited for environmental sound recognition.

---

## Dataset

**ADVANCE** (AuDio Visual Aerial sceNe reCognition datasEt)  
- 5,075 paired samples of geotagged aerial images and natural ambient audio  
- 13 scene categories: airport, sports land, beach, bridge, farmland, forest, grassland, harbor, lake, orchard, residential area, shrub land, train station  
- Split: 80% training / 20% validation

---

## Repository Structure

```
Audio-Visual-Retrieval/
├── Retrieval1/               # Main experiment directory
│   ├── datasets.py           # Dataset loading and preprocessing
│   ├── losses.py             # Triplet ranking loss
│   ├── evaluate.py           # Recall@K evaluation
│   └── *.py                  # Individual experiment scripts (models, training, testing)
└── .gitignore
```

> This is an ongoing project. New experiment scripts will be added to `Retrieval1/` over time. Each script is self-contained and corresponds to a specific model or experiment configuration.

---

## Models & Experiments

| Experiment | Visual Encoder | Audio Encoder | Extra Modules |
|---|---|---|---|
| Unimodal: AUD→AUD | — | CNN14 (frozen) | None |
| Unimodal: IMG→IMG | ResNet18 (frozen) | — | None |
| Exp 1: RN18 + CNN14 | ResNet18 (frozen) | CNN14 (frozen) | None |
| Exp 2: + Attention | ResNet18 (frozen) | CNN14 (frozen) | QDMVR + Coord. Attention |
| Exp 3: + Fine-tune | ResNet18 (partial) | CNN14 (partial) | None |
| Exp 4: CLIP + CNN14 | CLIP (frozen) | CNN14 (frozen) | None |
| Exp 5: CLIP + Fine-tune | CLIP (partial) | CNN14 (partial) | None |
| VSE++ | ResNet (frozen) | CNN14 (frozen) | Hard negatives |
| PVSE (K=2) | ResNet (frozen) | CNN14 (frozen) | Multi-embedding |
| AMFMN | ResNet + MVSA | CNN14 (frozen) | MVSA |
| **SARCI (ours)** | ResNet18 (frozen) | CNN14 (frozen) | QDMVR + ICLM |

---

## Key Results

**Cross-modal retrieval on the ADVANCE test set (Avg Recall@1):**

| Model | Avg R@1 | Avg R@5 | Avg R@10 |
|---|---|---|---|
| RN18 + CNN14 (frozen) | 26.94% | 61.92% | 71.69% |
| RN18 + CNN14 + Attention | 56.23% | 79.39% | 84.05% |
| VSE++ | 21.98% | 59.42% | 71.49% |
| AMFMN | 21.20% | 52.35% | 63.35% |
| **SARCI with CNN14** | **61.83%** | **85.03%** | **88.42%** |

SARCI with CNN14 significantly outperforms all baselines. The ICLM cross-modal interaction module and QDMVR attention are the key contributors to performance.

---

## Architecture

The adapted SARCI model has four components:

1. **Visual Encoder** — ResNet18 pretrained on ImageNet, with frozen weights
2. **Audio Encoder** — CNN14 pretrained on AudioSet, operating on log-mel spectrograms
3. **QDMVR** — Quaternion-attention Dominated Multiscale Visual Refinement; extracts and refines multiscale visual features using the SQA (Symmetric Quaternion Attention) mechanism
4. **ICLM** — Instruction-based Cross-Learning Module; performs cross-attention so each modality guides the other before similarity computation

Training uses a **hinge-based triplet ranking loss** with margin δ = 0.2, evaluated by **Recall@K** (K = 1, 5, 10) in both directions (IMG→AUD and AUD→IMG).

---

## Training Details

- Framework: PyTorch
- Optimizer: Adam, lr = 1e-4 with cosine scheduler
- Batch size: 32
- Max epochs: 50 (early stopping on R@5, patience = 10)
- Image size: 224×224, normalized with ImageNet stats
- Audio: resampled to 32kHz, 10s clips, 64 mel bins

---

## References

1. Y. Chen et al., "Scale-aware Adaptive Refinement and Cross-Interaction for Remote Sensing Audio-Visual Cross-Modal Retrieval," IEEE TGRS, 2024.
2. D. Hu et al., "Cross-task Transfer for Geotagged Audiovisual Aerial Scene Recognition," ECCV 2020.
3. Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks," IEEE/ACM TASLP, 2020.
4. F. Faghri et al., "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives," arXiv:1707.05612, 2017.
5. Y. Song and M. Soleymani, "Polysemous Visual-Semantic Embedding," CVPR 2019.
6. Z. Yuan et al., "AMFMN: Exploring a Fine-grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval," arXiv:2204.09868, 2022.
