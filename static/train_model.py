import argparse
import os
import sys
from os.path import join

import autogluon.core as ag
from autogluon.vision import ImagePredictor, ImageDataset

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


# The function returns the current path of the file.
def _current_path(*args):
    current_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(current_path, *args))


# The following function is used to parse different arguments.
def _parse_args():
    root_dir = _current_path(os.pardir)
    default_dataset_dir = join(root_dir, "static/datasets/dataset1")
    default_model_name = join(root_dir, "new_model.ag")
    default_output_dir = join(root_dir, "new_models")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory containing the X-ray dataset.")
    parser.add_argument('--model_name', type=str,
                        default=default_model_name,
                        help="Name to assign for the new model.")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory to store the new model.")
    args = parser.parse_args()
    return args


# Function to define the python paths.
def _init_python_path(args):
    os.makedirs(args.output_dir, exist_ok=True)


def main():
    # Parsing all arguments.
    args = _parse_args()
    _init_python_path(args)

    # Selecting the dataset to use.
    dataset_location = './static/datasets/' + args.dataset_dir
    print(dataset_location)
    # Loading the dataset.
    training_dataset, _, testing_dataset = ImageDataset.from_folders(dataset_location, train='train', test='test')
    print('train #', len(training_dataset), 'test #', len(testing_dataset))
    training_dataset.head()

    # Here we specify the models to use for the search space
    model_search_space = ag.Categorical('resnet18_v1b', 'mobilenetv3_small')

    # Here we manually specify certain parameters.
    batch_size = 8
    learning_rate = ag.Categorical(1e-2, 1e-3)

    hyperparameters = {'model': model_search_space, 'batch_size': batch_size, 'lr': learning_rate, 'epochs': 2}

    imagePredictor = ImagePredictor()  # Initializing the predictor.
    # Here we run the trials.
    imagePredictor.fit(training_dataset, time_limit=600 * 10, hyperparameters=hyperparameters,
                       hyperparameter_tune_kwargs={'num_trials': 2})
    print('Top-1 val acc: %.2f' % imagePredictor.fit_summary()['valid_acc'])

    # Using the test dataset to evaluate the model accuracy.
    testing_accuracy = imagePredictor.evaluate(testing_dataset)
    print('Test acc on hold-out static:', testing_accuracy)

    save_model_location = './new_models/' + args.model_name + '.ag'
    imagePredictor.save(save_model_location)
    print('========================================')
    print('Training has Finished, you can exit now.')
    print('========================================')


if __name__ == "__main__":
    main()
