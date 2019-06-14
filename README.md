## Conv-TasNet
A PyTorch implementation of Conv-TasNet described in ["TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation"](https://arxiv.org/abs/1809.07454).

## Results
| From  |  Activatoin |Norm | Causal | batch size |      #GPU      | SI-SDRi(dB) | SDRi(dB)|
|:-----:|:-----------:|:---:|:------:|:----------:|:--------------:|:-----------:|:-------:|
| Paper |   Softmax   | cLN |  Yes   |     -      |        -       |    10.8     |  11.2   |
| Here  |     ReLU    | cLN |  Yes   |     -      |  1 Tesla V100  |     -       |   -     |
| Here  |     ReLU    | cLN |  Yes   |     -      |  4 Tesla V100  |         |     |
| Paper |   Softmax   | gLN |  No    |     -      |        -       |    14.6     |  15.0   |
| Here  |   Softmax   | gLN |  No    |     8      |  2 GTX 1080TI  |    5.0      |  5.2    |
| Here  |   Sigmoid   | gLN |  No    |     8      |  2 GTX 1080TI  |    14.6     |  14.9   |
| Here  |     ReLU    | gLN |  No    |     8      |  2 GTX 1080TI  |    15.6     |  15.8   |
| Here  |     ReLU    | gLN |  No    |     20     |  4 Tesla V100  |    16.0     |  16.3   |

Seems increasing the `batch_size` will get better performance.

## Install
- PyTorch 0.4.1+
- Python3 (Recommend Anaconda)
- `pip install -r requirements.txt`

## Usage
```bash
scripts/run_tasnet.sh
```
