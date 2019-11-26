# DLTS - Deep Learning Assisted Heuristic Tree Search

This implementation was used to conduct the experiments in our paper [Deep Learning Assisted Heuristic Tree Search for the Container Pre-marshalling Problem](https://www.sciencedirect.com/science/article/pii/S0305054819302230)  (the preprint of the paper can be found [here](https://arxiv.org/abs/1709.09972)). Additionally to the code, this repository also contains the validation and test instances (and their optimal solutions) used in the paper.



## Dependencies

We evaluated DLTS using Python 3.5 and

- keras 1.1.0
- theano 0.8.2
- h5py 2.7.1

## Usage Example

To solve the 5x5_cv1 validation instances using the provided branching and bounding networks use the following command:

```bash
python deep_learn.py -t reference_solutions/5x5_cv1_validation -m pre-trained_models/pm_dnn_model_5x5.h5 -s -v pre-trained_models/pm_dnn_value_model_5x5.h5
```

