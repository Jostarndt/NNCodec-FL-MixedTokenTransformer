<div align="center">

# Efficient Federated Learning of Mixed-Token Transformers for Cellular Feature Prediction
  <img src="https://github.com/user-attachments/assets/4fba1aca-50ca-492f-901b-d601cc20874c" width="750" /> <br>
</div>


**Accepted at GLOBECOM2025!**

Currently only on ArXiv until official release.

## Short Paper Abstract
We propose Mixed-Token Transformers (MTT) for predicting mobile network features in Autonomous Networks using Federated Learning. Our approach enables multiple network cells across five geographically distinct regions to collaboratively learn while preserving privacy. Using NNCodec compression, we reduce FL communication overhead to below 1% with negligible performance loss, while achieving ~5× faster inference on the Berlin V2X dataset.

## Citation

```
ArXiv bibtex
```

## Installation

Requirements

- Python >= 3.10

install 

```bash
pip install -e . 
```

For tokenization, additionally

```bash
pip install pyarrow
```


## Usage

### Tokenization
To Tokenize the data, download the cellular_dataframe.parquet file from [Berlin V2X](https://ieee-dataport.org/open-access/berlin-v2x). Then run one of the two preprocessing steps

  A. Tokenize for the digitwise transformer: 

  ```
  python3 telko_dataloader.py pretokenize_telko_digit --data_path ./cellular_dataframe.parquet
  ```

  B. Tokenize for the mixed token transformer

```
python3 telko_dataloader.py pretokenize_telko_mtt --data_path ./cellular_dataframe.parquet
```

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `--data_path` | Path to `cellular_dataframe.parquet` |

### Training

```bash
python3 example/nnc_fl.py --dataset=V2X --dataset_path=./preprocessing/ --model=LSTM --model_rand_int --num_clients=5 --epochs=30 --batch_size=8 --max_batches=300 --max_batches_test=150 --TLM_size=1 --tokenizer_path=./tokenizer/telko_tokenizer.model
```


**Parameters**

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Dataset to use (V2X) |
| `--dataset_path` | Path to preprocessed data |
| `--model` | Model architecture, one of (DBD/mtt/LSTM) |
| `--model_rand_int` | Enable model randomization |
| `--num_clients` | Number of federated clients |
| `--epochs` | Training epochs per round |
| `--batch_size` | Batch size for training |
| `--max_batches` | Max batches for training |
| `--max_batches_test` | Max batches for testing |
| `--TLM_size` | Model size, one of (1,2) |
| `--tokenizer_path` | Path to tokenizer model |
| `--wandb`| Log on WandB |
| `--wandb_key="my_key"`|WandB key (optional)|
|`--wandb_run_name="my_project"`| WandB project (optional)|


### Evaluation
```
python3 eval.py --model_path=best_mtt_MTT_UC_slurmID_67554_gitID_917ac77.pt --batch_size=1 --dataset=V2X --dataset_path=./preprocessing/ --model=mtt --TLM_size=1 --tokenizer_path=./tokenizer/telko_tokenizer.model --max_seq_len=528 --workers=0  --spec_feat_test="datarate"
```

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to model "file.pt" |
| `--batch_size` | batch size |
| `--max_seq_len` | Maximum sequence length |
| `--workers` |  |
| `--spec_feat_test="feature"` | Specific feature "feature" to test on, without accumulated losses, e.g. "datarate" |


    


  The pre-tokenized [Berlin V2X dataset](https://ieee-dataport.org/open-access/berlin-v2x) can be downloaded here: https://datacloud.hhi.fraunhofer.de/s/CcAeHRoWRqe5PiQ
  and the pre-trained Sentencepiece Tokenizer is included in this repository at [telko_tokenizer.model](https://github.com/d-becking/nncodec2/blob/master/example/tokenizer/).
  
  Resulting bitstreams and the best performing global TLM of all communication rounds will be stored in a `results` directory (with path set via `--results`).
  To evaluate this model, execute:

  ```bash
  python example/eval.py --model_path=<your_path>/best_tinyllama_.pt --batch_size=1 --dataset=V2X \
  --dataset_path=<your_path>/v2x --model=tinyllama --TLM_size=1 --tokenizer_path=./example/tokenizer/telko_tokenizer.model
  ```







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

The default configuration launches FL with two _ResNet-56_ clients learning the _CIFAR-100_ classification task. The _CIFAR_ dataset
is automatically downloaded if not available under `--dataset_path` (~170MB).
```bash
python example/nnc_fl.py --dataset_path=<your_path> --model_rand_int --epochs=30 --compress_upstream --compress_downstream --err_accumulation --compress_differences
```



## Important References / EUCnc

  The pre-tokenized [Berlin V2X dataset](https://ieee-dataport.org/open-access/berlin-v2x) can be downloaded here: https://datacloud.hhi.fraunhofer.de/s/CcAeHRoWRqe5PiQ
  and the pre-trained Sentencepiece Tokenizer is included in this repository at [telko_tokenizer.model](https://github.com/d-becking/nncodec2/blob/master/example/tokenizer/).
  
  Resulting bitstreams and the best performing global TLM of all communication rounds will be stored in a `results` directory (with path set via `--results`).
  To evaluate this model, execute:


    

## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of the use of the contents of this repository.

For more information and bug reports, please contact: [nncodec@hhi.fraunhofer.de](mailto\:nncodec@hhi.fraunhofer.de)

**Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.**

**All rights reserved.**
