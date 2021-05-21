# Attention-Based Contextual Language Modeling Adaptation

This project provides the source to reproduce the main methods of the paper
"Attention-Based Contextual Language Model Adaptation for Speech Recognition",
submitted to ACL 2021. This codebase also implements additional functionality
that was not explicitly described in the paper, such as experimental methods
for combining multiple types of non-linguistic context together (e.g. geo-location,
and datetime).

## Onboarding

Basic environment setup

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

## Data

We are unable to provide the data we used for the results we report in our
paper. However, to illustrate the expected input by the model we provide data
samples in the data folder that illustrate the data format. In general, data
should be structured in a tsv format where the first column corresponds to
the transcribed utterance and the subsequent columns correspond to associated
non-linguistic context.

## Training a Model

In all of our experiments, we adapt a base 1-layer LSTM model with additional
context, using an attention mechanism. To run an experiment, you need to
define a config file with the desired configurations for the model architecture,
data processing, model training and model evaluation parameters. To illustrate
how to setup a config file for an experiment, we provide a sample config file
under experiments/demo. The sample config provides the configurations for
conditioning an NLM on datetime context, using a bahdanau attention mechanism.

To train a model using the sample config, run the following command from the
root directory.

```
python3 run_model.py experiments/demo/train_config.ini
```

Running this script will generate a log containing the training results. Using
the provided train_config.ini config, you should expect the see the following
final evaluation (numbers might vary a bit):

```
Finished Evaluation Model.
  Full Dev Data -- Loss: 4.678730704567649, PPL: 107.63334655761719
  Head Dev Data -- Loss: 4.679276899857954, PPL: 107.69217681884766
  Tail Dev Data -- Loss: 4.678184509277344, PPL: 107.57459259033203
```

## Running Inference

Similarly to how we train a model by defining a config file, we also provide a
config file for evaluating a model on a given dataset. In experiment/demo you
will find a sample config for evaluating the same model defined in train_config.ini,
using the sample dev dataset we provide in data/dev.

To run inference, run the following command from the root directory.

```
python3 run_inference.py experiments/demo/test_config.ini
```

Note that the configuration we provide in test_config assumes that you have
already trained a model using the config in train_config.ini.


For additional information or questions, reach out to mrtimri@amazon.com
