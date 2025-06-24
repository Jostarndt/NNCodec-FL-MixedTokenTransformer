<div align="center">

![nncodec_logo](https://github.com/d-becking/nncodec-icml-2023-demo/assets/56083075/f310c92e-537e-4960-b223-7ec51c9accc3)

# A Software Implementation of the Neural Network Coding (NNC) Standard [ISO/IEC 15938-17] 

</div>


## Table of Contents
- [Information](#information)
- [Installation](#installation)
- [NNCodec Usage](#nncodec-usage)
  * [Paper results](#paper-results)
- [Citation and Publications](#citation-and-publications)
- [License](#license)

## Information

This repository hosts a beta version of NNCodec 2.0, which incorporates new compression tools for incremental neural 
network data, as introduced in the second edition of the NNC standard. Additionally, it features a pipeline for coding 
"Tensors in AI-based Media Processing" to address recent MPEG requirements for coding individual tensors rather than 
entire neural networks or differential updates to a base neural network.

The repository also includes a novel use case example demonstrating federated learning for tiny language models in a telecommunications application.

The official NNCodec 1.0 git repository, which served as the foundation for this project, can be found here:

[![Conference](https://img.shields.io/badge/fraunhoferhhi-nncodec-green)](https://github.com/fraunhoferhhi/nncodec)

It also contains a [Wiki-Page](https://github.com/fraunhoferhhi/nncodec/wiki) providing further information on NNCodec. 

Upon approval, this second version will update the official Git repository.

### The Fraunhofer Neural Network Encoder/Decoder (NNCodec)
The Fraunhofer Neural Network Encoder/Decoder Software (NNCodec) is an efficient implementation of NNC ([Neural Network Coding ISO/IEC 15938-17](https://www.iso.org/standard/85545.html)), 
which is the first international standard for compressing (incremental) neural network data.

NNCodec provides an encoder and decoder with the following main features:
- Standard-compliant implementation of the core compression technologies, including, e.g., DeepCABAC, quantization, and sparsification
- User-friendly interface
- Built-in support for common deep learning frameworks (e.g., PyTorch)
- Integrated support for data-driven compression tools on common datasets (ImageNet, CIFAR, PascalVOC)
- Built-in support for [Flower](https://flower.ai), a prominent and widely used Federated AI framework
- Separate pipelines for Neural Network (NN) Coding, Tensor Coding, and Federated Learning


## Installation

### Requirements

- python >= 3.8 with working pip
- **Windows:** Microsoft Visual Studio 2015 Update 3 or later

### Package installation

NNCodec V2 supports pip installation:

```bash
pip install nncodec
```

After installation the software can be used by importing the main module:
```python
import nncodec
```

## NNCodec Usage
![NNCodec2](https://github.com/user-attachments/assets/564b9d02-a706-459a-a8bb-241d2ec4608f)

### [TBD]

#### Logging (comparative) results using Weights & Biases

We used Weights & Biases (wandb) for experiment logging. Enabling `--wandb` also enables Huffman and bzip2 encoding of the data payloads and the calculation of the Shannon entropy. If you want to use it, add your wandb key and optionally an experiment identifier for the run (--wandb_run_name).

```shell
--wandb, --wandb_key, --wandb_run_name
```

## Paper results


- ### EuCNC 2025 Poster Session
    We present **"Efficient Federated Learning Tiny Language Models for Mobile Network Feature Prediction"** at the Poster Session I of the 2025 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit).

    **TL;DR** -  This work introduces a communication-efficient Federated Learning (FL) framework for training tiny language models (TLMs) that collaboratively learn to predict mobile network features (such as ping, SNR or frequency band) across five geographically distinct regions from the Berlin V2X dataset. Using NNCodec, the framework reduces communication overhead by over 99% with minimal performance degradation, enabling scalable FL deployment across autonomous mobile network cells.

    <img src="https://github.com/user-attachments/assets/4fba1aca-50ca-492f-901b-d601cc20874c" width="750" /> <br>

    The codebase for reproducing experimental results and evaluating NNCodec in an FL environment is available here:

    [![Conference](https://img.shields.io/badge/EuCNC-Paper-blue)](https://arxiv.org/abs/2504.01947)



- ### ICML 2023 Neural Compression Workshop
    Our paper titled **"NNCodec: An Open Source Software Implementation of the Neural Network Coding 
ISO/IEC Standard"** was awarded a Spotlight Paper at the ICML 2023 Neural Compression Workshop.

    **TL;DR** -  The paper presents NNCodec, analyses its coding tools with respect to the principles of information theory and gives comparative results for a broad range of neural network architectures. 

    The code for reproducing the experimental results of the paper and a software demo are available 
here:

    [![Conference](https://img.shields.io/badge/ICML-Paper-blue)](https://openreview.net/forum?id=5VgMDKUgX0)


## Citation and Publications
If you use NNCodec in your work, please cite:
```
@inproceedings{becking2023nncodec,
title={{NNC}odec: An Open Source Software Implementation of the Neural Network Coding {ISO}/{IEC} Standard},
author={Daniel Becking and Paul Haase and Heiner Kirchhoffer and Karsten M{\"u}ller and Wojciech Samek and Detlev Marpe},
booktitle={ICML 2023 Workshop Neural Compression: From Information Theory to Applications},
year={2023},
url={https://openreview.net/forum?id=5VgMDKUgX0}
}
```
### Publications (chronological order)
- D. Becking et al., **"Neural Network Coding of Difference Updates for Efficient Distributed Learning Communication"**, IEEE Transactions on Multimedia, vol. 26, pp. 6848–6863, 2024, doi: 10.1109/TMM.2024.3357198, Open Access
- D. Becking et al. **"NNCodec: An Open Source Software Implementation of the Neural Network Coding ISO/IEC Standard"**, 40th International Conference on Machine Learning (ICML), 2023, Neural Compression Workshop (Spotlight)
- H. Kirchhoffer et al. **"Overview of the Neural Network Compression and Representation (NNR) Standard"**, IEEE Transactions on Circuits and Systems for Video Technology, pp. 1-14, July 2021, doi: 10.1109/TCSVT.2021.3095970, Open Access
- P. Haase et al. **"Encoder Optimizations For The NNR Standard On Neural Network Compression"**, 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 3522-3526, doi: 10.1109/ICIP42928.2021.9506655.
- K. Müller et al. **"Ein internationaler KI-Standard zur Kompression Neuronaler Netze"**, FKT- Fachzeitschrift für Fernsehen, Film und Elektronische Medien, pp. 33-36, September 2021
- S. Wiedemann et al., **"DeepCABAC: A universal compression algorithm for deep neural networks"**, in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2020.2969554.

## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of the use of the contents of this repository.

For more information and bug reports, please contact: nncodec@hhi.fraunhofer.de

**Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.**

**All rights reserved.**
