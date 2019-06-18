## Conv-TasNet
A PyTorch implementation of Conv-TasNet described in ["TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation"](https://arxiv.org/abs/1809.07454).

## Results
|   From  | Activatoin |  Norm | Causal | batch size |      #GPU      | SI-SDRi(dB) | SDRi(dB)|
|:-------:|:----------:|:-----:|:------:|:----------:|:--------------:|:-----------:|:-------:|
|**Paper**| **Softmax**|**cLN**|**Yes** |     -      |        -       |   **10.8**  | **11.2**|
|   Mine  |     ReLU   |  cLN  |  Yes   |     6      |  1 Tesla V100  |     11.3    |  11.6     |
|   Mine  |     ReLU   |  cLN  |  Yes   |     24     |  4 Tesla V100  |   **11.2**  | **11.5**|
|**Paper**| **Softmax**|**cLN**| **No** |     -      |        -       |   **14.0**  | **14.4**|
|   Mine  |     ReLU   |  cLN  |  No    |     24     |  4 Tesla V100  |    15.1     |  15.4   |
|**Paper**| **Softmax**|**gLN**| **No** |     -      |        -       |   **14.6**  | **15.0**|
|   Mine  |   Softmax  |  gLN  |  No    |     8      |  2 GTX 1080TI  |    5.0      |  5.2    |
|   Mine  |   Sigmoid  |  gLN  |  No    |     8      |  2 GTX 1080TI  |    14.6     |  14.9   |
|   Mine  |     ReLU   |  gLN  |  No    |     8      |  2 GTX 1080TI  |    15.6     |  15.8   |
|   Mine  |     ReLU   |  gLN  |  No    |     20     |  4 Tesla V100  |   **16.0**  | **16.3**|

Seems increasing the `batch_size` will get better performance for non-causal convolution. `SI-SDR` and `SI-SNR` are the same thing (different name) in different papers.

## Install
- PyTorch 0.4.1+
- Python3 (Recommend Anaconda)
- `pip install -r requirements.txt`

## Usage
```bash
scripts/run_tasnet.sh
```
