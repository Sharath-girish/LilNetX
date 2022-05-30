# LilNetX
Official PyTorch implementation of LilNetX: Lightweight Networks with EXtreme Model Compression and Structured Sparsification


## Installation
Create new virtualenv/conda environment and run the following command
```
pip install -r requirements.txt
```
To run ImageNet experiments with the faster FFCV implementation, follow instructions on https://github.com/libffcv/ffcv to install the FFCV library.

## Example commands
Hyperparameters are defined in yaml files in the [configs](configs/) folder. An example run command for training a ResNet20 with width 4 on the Cifar10 dataset would look like
```
python main.py --config configs/cifar10_resnet20.yaml
```
Hyperparameters can additionally be overriden with command-line arguments which are dot separated. For e.g. training the same model as above but with train batch size of 512 (instead of the default 256) would look like
```
python main.py --config configs/cifar10_resnet20.yaml --trainer.train_batch 512
```

## License

This project is released under the MIT License. Please review the [License file](LICENSE) for more details.
