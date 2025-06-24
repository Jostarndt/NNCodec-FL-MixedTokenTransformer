<div align="center">

<img src="https://github.com/fraunhoferhhi/nncodec/assets/65648299/69b41b38-19ed-4c45-86aa-2b2cd4d835f7" width="660"/>

# A Software Implementation of the ISO/IEC 15938-17 Neural Network Coding (NNC) Standard

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
- Built-in support for [*Flower*](https://flower.ai), a prominent and widely used Federated AI framework
- Separate pipelines for Neural Network (NN) Coding, Tensor Coding, and Federated Learning


## Installation

### Requirements

- python >= 3.8 with working pip
- **Windows:** Microsoft Visual Studio 2015 Update 3 or later

### Package installation

NNCodec 2.0 supports pip installation:

```bash
pip install nncodec
```

After installation, the software can be used by importing the main module:
```python
import nncodec
```

## NNCodec Usage
<div align="center">
<img src="https://github.com/user-attachments/assets/564b9d02-a706-459a-a8bb-241d2ec4608f" width="660"/>
</div>

NNCodec 2.0, as depicted above, includes three main pipelines:
- One for tensorial data in AI-based media processing (e.g., function coefficients, feature maps, ...),
  ```python
  from nncodec.tensor import encode, decode
  ```
- one for coding entire neural networks (or their differential updates), and
  ```python
  from nncodec.nn import encode, decode
  ```
- one for federated learning scenarios.
  ```python
  from nncodec.fl import NNClient, NNCFedAvg
  ```

### Coding Tensors in AI-based Media Processing (TAIMP)

The [tensor_coding.py](https://github.com/d-becking/nncodec2/blob/master/example/tensor_coding.py) script provides
encoding and decoding examples of random tensors.
The first example codes a random tensor (which could also be an integer tensor):

```python
example_tensor = torch.randn(256, 64, 64) # torch.randint(0, 255, (3, 3, 32, 32)) # example 8-bit uint tensor
bitstream = encode(example_tensor, args_dict) 
dec_tensor = torch.tensor(decode(bitstream, args_dict["tensor_id"]), device=device, dtype=torch.float32)
```
here, `args_dict` is a python dictionary that specifies the encoding configuration. The default configuration is:

```python
args_dict = { 'approx_method': 'uniform', # Quantization method [uniform or codebook]
              'qp': -32, # main quantization parameter (QP)
              'nonweight_qp': -75, # QP for non-weights, e.g., 1D or BatchNorm params (default: -75, i.e., fine quantization)
              'use_dq': True, # enables dependent scalar / Trellis-coded quantization
              'bitdepth': None, # Optional: integer-aligned bitdepth for limited precision [1, 31] bit; note: overwrites QPs.
              'quantize_only': False, # if True encode() returns quantized parameter instead of bitstream 
              'tca': False, # enables Temporal Context Adaptation (TCA)
              'row_skipping': True, # enables skipping tensor rows from arithmetic coding if entirely zero
              'sparsity': 0.0, # introduces mean- & std-based unstructured sparsity [0.0, 1.0]
              'struct_spars_factor': 0.0, # introduces structured per-channel sparsity (based on channel means); requires sparsity > 0.0
              'job_identifier': 'TAIMP_coding', # Name extension for generated *.nnc bitstream files and for logging
              'results': '.', # path where results / bitstreams shall be stored
              'tensor_id': '0', # identifier for tensor
              'tensor_path': None, # path to tensor to be encoded
              'compress_differences': False, # if True bitstream represents a differential update of a base tensor; set automatically if TCA enabled
              'verbose': True # print stdout process information.
             }
```

The second example — if tensor_coding.py run with `--incremental` — updates 50% of the example tensor's elements for `num_increments`
iterartions and stores the previously decoded collocated tensor in `approx_param_base`, to use it for Temporal Context Adaptation (TCA).

`approx_param_base` must be initialized with 
```python
approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}} 
```

### Coding Neural Networks and Differential Neural Network Updates
[TBD]

### Federated Learning with NNCodec

The [nnc_fl.py](https://github.com/d-becking/nncodec2/blob/master/example/nnc_fl.py) file implements a base script for communication-efficient
Federated Learning with NNCodec. It imports the `NNClient` and `NNCFedAvg` classes — specialized NNC-[*Flower*](https://flower.ai) objects — that 
are responsible for establishing and handling the compressed FL environment.

The default configuration launches FL with two ResNet-56 clients learning the CIFAR-100 classification task. The CIFAR dataset
is automatically downloaded if not available under `--dataset_path` (~170MB).
```bash
python example/nnc_fl.py --dataset_path=<your_path> --model_rand_int --epochs=30 --compress_upstream --compress_downstream --err_accumulation --compress_differences
```

Main coding tools and hyperparameter settings for coding are:
```bash
--qp   'Quantization parameter (QP) for NNs (default: -32)'
--diff_qp   'Quantization parameter for dNNs. Defaults to QP if unspecified (default: None)'
--nonweight_qp  'QP for non-weights, e.g., 1D or fBatchNorm params (default: -75)'
--opt_qp  'Enables layer-wise QP modification based on relative layer size within NN'
--use_dq  'Enables dependent scalar / Trellis-coded quantization'
--bitdepth 'Optional: integer-aligned bitdepth for limited precision [1, 31] bit; note: overwrites QPs.'
--bnf  'Enables incremental BatchNorm Folding (BNF)'
--sparsity  'Introduces mean- & std-based unstructured sparsity [0.0, 1.0] (default: 0.0)'
--struct_spars_factor 'Introduces structured per-channel sparsity (based on channel means); requires sparsity > 0 (default: 0.9)'
--row_skipping  'Enables skipping tensor rows from arithmetic coding that are entirely zero'
--tca 'Enables Temporal Context Adaptation (TCA)'
```

Additional important hyperparameters for FL (among others in [nnc_fl.py](https://github.com/d-becking/nncodec2/blob/master/example/nnc_fl.py)):
```bash
--compress_differences  'Weight differences wrt. to base model (dNN) are compressed, otherwise full base models (NN) are communicated'
--model_rand_int 'If set, model is randomly initialized, i.e., w/o loading pre-trained weights'
--num_clients 'Number of clients in FL scenario (default: 2)'
--compress_upstream 'Compression of clients-to-server communication'
--compress_downstream 'Compression of server-to-clients communication'
--err_accumulation  'If set, quantization errors are locally accumulated ("residuals") and added to NN update prior to compression'
```
Section [Paper results](#EuCNC-2025-Poster-Session) (EuCNC) below introduces an additional use case and implementation of NNCodec 2.0 FL with tiny language models collaboratively learning feature predictions in cellular data.

### Logging results using Weights & Biases

We used Weights & Biases (wandb) for experimental results logging. Enable `--wandb` if you want to use it. Add your wandb key and optionally an experiment identifier for the run:

```bash
--wandb, --wandb_key="my_key", --wandb_run_name="my_project"
```

## Paper results


- ### EuCNC 2025 Poster Session  
  [![Conference](https://img.shields.io/badge/EuCNC-Paper-blue)](https://arxiv.org/abs/2504.01947)

  We presented **"Efficient Federated Learning Tiny Language Models for Mobile Network Feature Prediction"** at the Poster Session I of the 2025 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit).
    
  **TL;DR** -  This work introduces a communication-efficient Federated Learning (FL) framework for training tiny language models (TLMs) that collaboratively learn to predict mobile network features (such as ping, SNR or frequency band) across five geographically distinct regions from the Berlin V2X dataset. Using NNCodec, the framework reduces communication overhead by over 99% with minimal performance degradation, enabling scalable FL deployment across autonomous mobile network cells.
  <img src="https://github.com/user-attachments/assets/4fba1aca-50ca-492f-901b-d601cc20874c" width="750" /> <br>

  To reproduce the experimental results and evaluate NNCodec in the telco FL setting described above, execute:

  ```bash
  python example/nnc_fl.py --dataset=V2X --dataset_path=<your_path>/v2x --model=tinyllama --model_rand_int --num_clients=5 --epochs=30 --compress_upstream --compress_downstream --err_accumulation --compress_differences --qp=-18 --batch_size=8 --max_batches=300 --max_batches_test=150 --sparsity=0.8 --struct_spars_factor=0.9 --TLM_size=1 --tca --tokenizer_path=./example/tokenizer/telko_tokenizer.model
  ```

  The pre-tokenized [Berlin V2X dataset](https://ieee-dataport.org/open-access/berlin-v2x) can be downloaded here: https://datacloud.hhi.fraunhofer.de/s/CcAeHRoWRqe5PiQ
  and the pretrained Sentencepiece Tokenizer is included in this repository at [telko_tokenizer.model](https://github.com/d-becking/nncodec2/blob/master/example/tokenizer/).
  
  Resulting bitstreams and the best performing global TLM of all communication rounds will be stored in a `results` directory (with path set via `--results`).
  Evaluating this model is possible by executing:

  ```bash
  python example/eval.py --model_path=<your_path>/best_tinyllama_.pt --batch_size=1 --dataset=V2X --dataset_path=<your_path>/v2x --model=tinyllama --TLM_size=1 --tokenizer_path=./example/tokenizer/telko_tokenizer.model
  ```


- ### ICML 2023 Neural Compression Workshop
  [![Conference](https://img.shields.io/badge/ICML-Paper-blue)](https://openreview.net/forum?id=5VgMDKUgX0)

  Our paper titled **"NNCodec: An Open Source Software Implementation of the Neural Network Coding 
  ISO/IEC Standard"** was awarded a Spotlight Paper at the ICML 2023 Neural Compression Workshop.

    **TL;DR** -  The paper presents NNCodec 1.0, analyses its coding tools with respect to the principles of information theory and gives comparative results for a broad range of neural network architectures.
    The code for reproducing the experimental results of the paper and a software demo are available 
here:
  
  [![Conference](https://img.shields.io/badge/ICML-Code-red)](https://github.com/d-becking/nncodec-icml-2023-demo)
    


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
