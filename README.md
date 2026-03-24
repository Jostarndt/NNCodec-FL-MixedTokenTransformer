<div align="center">

# Efficient Federated Learning of Mixed-Token Transformers for Cellular Feature Prediction
  <img src="https://github.com/user-attachments/assets/4fba1aca-50ca-492f-901b-d601cc20874c" width="750" /> <br>
</div>


**Accepted at GLOBECOM2025!**

PDF available under  [ieeexplore.ieee.org/document/11431868](https://ieeexplore.ieee.org/document/11431868).

## Short Paper Abstract
We propose Mixed-Token Transformers (MTT) for predicting mobile network features in Autonomous Networks using Federated Learning. Our approach enables multiple network cells across five geographically distinct regions to collaboratively learn while preserving privacy. Using NNCodec compression, we reduce FL communication overhead to below 1% with negligible performance loss, while achieving ~5× faster inference on the Berlin V2X dataset.


## Installation

Requirements

- Python >= 3.10

install 

```bash
pip install -e . 
```


## Usage

### Tokenization
To Tokenize the data, download the cellular_dataframe.parquet file from [Berlin ](https://ieee-dataport.org/open-access/berlin-v2x). Then run one of the two preprocessing steps

  A. Tokenize for the digitwise transformer: 

  ```
  python3 example/preprocessing/telko_dataloader.py pretokenize_telko_digit --data_path ./path_to_/cellular_dataframe.parquet
  ```

  B. Tokenize for the mixed token transformer

```
python3 example/preprocessing/telko_dataloader.py pretokenize_telko_mtt --data_path ./path_to_/cellular_dataframe.parquet
```

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `--data_path` | Path to `cellular_dataframe.parquet` |

### Training

```bash
python3 example/nnc_fl.py --dataset_path=./example/preprocessing/output --model=mtt --num_clients=5 --epochs=30 --batch_size=8 --max_batches=300  --TLM_size=1 --tokenizer_path=./example/tokenizer/telko_tokenizer.model
```


**Parameters**

| Parameter | Description |
|-----------|-------------|
| `--dataset_path` | Path to preprocessed data |
| `--model` | Model architecture, one of (DBD/mtt/LSTM) |
| `--num_clients` | Number of federated clients |
| `--epochs` | Training epochs per round |
| `--batch_size` | Batch size for training |
| `--max_batches` | Max batches for training |
| `--TLM_size` | Model size, one of (1,2) |
| `--tokenizer_path` | Path to tokenizer model |
| `--wandb`| Log on WandB |
| `--wandb_key="my_key"`|WandB key (optional)|
|`--wandb_run_name="my_project"`| WandB project (optional)|

Resulting bitstreams and the best performing global TLM of all communication rounds will be stored in a `results` directory (with path set via `--results`). Evaluation can be found in the next section.


### Evaluation
```
python3 example/eval.py --model_path=results/best_mtt_.pt --batch_size=1 --dataset_path=./example/preprocessing/output --model=mtt --TLM_size=1 --tokenizer_path=./example/tokenizer/telko_tokenizer.model --workers=0
```

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to model "file.pt" |
| `--max_seq_len` | Maximum sequence length |
| `--workers` | Number of workers |
| `--spec_feat_test=feature` | Specific feature "feature" to test on, without accumulated losses, e.g. "datarate" |


    


  The pre-tokenized [Berlin V2X dataset](https://ieee-dataport.org/open-access/berlin-v2x) can be downloaded here: https://datacloud.hhi.fraunhofer.de/s/CcAeHRoWRqe5PiQ
  and the pre-trained Sentencepiece Tokenizer is included in this repository at [telko_tokenizer.model](https://github.com/d-becking/nncodec2/blob/master/example/tokenizer/).
  





## NNCodec
<div align="center">
<img src="https://github.com/user-attachments/assets/564b9d02-a706-459a-a8bb-241d2ec4608f" width="660"/>
</div>

This repository is based on NNCodec 2.0, which incorporates new compression tools for incremental neural 
network data, as introduced in the second edition of the NNC standard. It also supports coding 
"Tensors in AI-based Media Processing" (TAIMP), addressing recent MPEG requirements for coding individual tensors rather 
than entire neural networks or differential updates to a base neural network.


NNCodec is an efficient implementation of NNC ([Neural Network Coding ISO/IEC 15938-17](https://www.iso.org/standard/85545.html)), 
the first international standard for compressing (incremental) neural network data.

The [NNCodec 2.0](https://github.com/d-becking/nncodec2/), as depicted above, includes a pipeline for federated learning scenarios on which our work is based on. The Federated AI is based on [*Flower*](https://flower.ai), a prominent and widely used framework

  ```python
  from nncodec.fl import NNClient, NNCFedAvg
  ```


### Federated Learning with NNCodec

The original file [nnc_fl.py](https://github.com/d-becking/nncodec2/blob/master/example/nnc_fl.py) implements a base script for communication-efficient
Federated Learning with NNCodec. It imports the `NNClient` and `NNCFedAvg` classes — specialized NNC-[*Flower*](https://flower.ai) objects — that 
are responsible for establishing and handling the compressed FL environment.

```bash
python3 example/nnc_fl.py --dataset_path=./example/preprocessing/output --model=mtt --num_clients=5 --epochs=30 --batch_size=8 --max_batches=300  --TLM_size=1 --tokenizer_path=./example/tokenizer/telko_tokenizer.model --compress_upstream --compress_downstream --err_accumulation --compress_differences --qp=-24 --tca --sparsity=0.5 --struct_spars_factor=0.9 --row_skipping
```

**Compression Parameters**

| Parameter | Description |
|-----------|-------------|
| `--compress_upstream` | Compression of clients-to-server communication |
| `--compress_downstream` | Compression of server-to-clients communication |
| `--err_accumulation` | Quantization errors are locally accumulated ("residuals") and added to NN update prior to compression |
| `--compress_differences` | Weight differences (dNN) wrt. a base model are compressed, otherwise full base models (NN) are communicated |
| `--qp` | Quantization parameter (larger is coarser) |
| `--tca` | Enables Temporal Context Adaptation (TCA) |
| `--sparsity` | Introduces mean- & std-based unstructured sparsity [0.0, 1.0] |
| `--struct_spars_factor` | Introduces structured per-channel sparsity (based on channel means); requires --sparsity > 0 |
| `--row_skipping`| Enables skipping tensor rows in arithmetic coding stage if they are entirely zero |



## Important References

  The pre-tokenized [Berlin V2X dataset](https://ieee-dataport.org/open-access/berlin-v2x) can be downloaded here: https://datacloud.hhi.fraunhofer.de/s/CcAeHRoWRqe5PiQ
  and the pre-trained Sentencepiece Tokenizer is included in this repository at [telko_tokenizer.model](https://github.com/d-becking/nncodec2/blob/master/example/tokenizer/).
  
  Resulting bitstreams and the best performing global TLM of all communication rounds will be stored in a `results` directory (with path set via `--results`).

    
## Citation

```
@INPROCEEDINGS{nnc-fl-mtt,
  author={Becking, Daniel and Arndt, Jost and Friese, Ingo and Müller, Karsten and Ma, Jackie and Buchholz, Thomas and Galkow-Schneider, Mandy and Samek, Wojciech and Marpe, Detlev},
  booktitle={GLOBECOM 2025 - 2025 IEEE Global Communications Conference}, 
  title={Efficient Federated Learning of Mixed-Token Transformers for Cellular Feature Prediction}, 
  year={2025},
  volume={},
  number={},
  pages={2643-2649},
  doi={10.1109/GLOBECOM59602.2025.11431868}}
```


## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of the use of the contents of this repository.

For more information and bug reports, please contact: [nncodec@hhi.fraunhofer.de](mailto\:nncodec@hhi.fraunhofer.de)

**Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.**

**All rights reserved.**
