# PointNet for 3D Object Classification (PyTorch)

This repository contains an implementation of PointNet for 3D object classification in PyTorch, based on the paper ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"](https://arxiv.org/abs/1612.00593) by Charles R. Qi et al.

## Dataset

The model is trained and tested on the [ModelNet10](https://modelnet.cs.princeton.edu/) dataset, which contains 10 categories of objects (bed, chair, dresser, etc.) with 4899 objects in total.

## Requirements

The code requires the following libraries:

- PyTorch
- NumPy
- open3d
- scikit-learn
- matplotlib
- plotly
- tqdm

These can be installed using pip:

pip install torch numpy open3d scikit-learn matplotlib plotly tqdm

## Training

To train the model, run:

python train.py --data_path /path/to/dataset --batch_size 32 --num_epochs 50 --learning_rate 0.001

You can adjust the batch size, number of epochs, and learning rate as needed.

## Evaluation

To evaluate the trained model on the test set, run:

python test.py --data_path /path/to/dataset --model_path /path/to/model


Replace `/path/to/dataset` with the path to the ModelNet10 dataset and `/path/to/model` with the path to the saved model checkpoint.

## Results

The model achieves an accuracy of 90.4% on the test set.

## References

- Qi, Charles R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017.
- ModelNet10 Dataset: https://modelnet.cs.princeton.edu/

