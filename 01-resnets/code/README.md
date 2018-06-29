# ResNet-style Models

Example code re-implementing a few ResNet-style models.

## Usage

### Creating Conda environment
  1. Make sure you have [Anaconda or Miniconda](https://conda.io/docs/download.html)
  2. Download repo and `cd` into it. Run `conda env create --file=environment.yml`.
  3. Add CUDA to the environment if you have a GPU.
  4. (Optional) Add interpreter to PyCharm
    - Go to `Preferences > Project > Project Interpreter`
    - Click the gear icon, then 'Add'
    - Select 'Conda Environment' and 'Existing Environment'
    - Click the three dots, then find the interpreter. Should be somewhere like
    `/Users/christopherchute/anaconda3/envs/res/bin/python`.

### Train (Dummy on CIFAR=10)
  1. Get a GPU instance, create and activate conda environment as described above.
  2. Run `python train.py --name=res`.

### TensorBoard
  1. `cd` into this project's root directory.
  2. Run `tmux new -s tb` for  a new `tmux` session named `tb`.
  3. Run `source activate res` to set up the virtual environment.
  4. Run `tensorboard --logdir=. --port=5678`.
  5. Hit `ctrl-b`, `d` to detach from the `tmux` session. Later run `tmux a -t train` to re-attach.
  6. (Local) Run `ssh -N -f -L localhost:1234:localhost:5678 <remote_host>`.
  7. (Local) In a web browser, go to `http://localhost:1234`.

## Contents
```text
+ args: Command-line arg parsing
+ data: Placeholder for CIFAR dataset
+ data_loader: Wraps CIFAR data loader
+ logger: Logs training info to the console and TensorBoard
+ models: Example code for ResNet-style models to train on CIFAR-10
  - resnet.py: Original ResNet
  - densenet.py: DenseNet
  - resnext.py: ResNeXt
  - se_resnext.py: ResNeXt with Squeeze-and-Excitation blocks
+ saver: Saves and loads model checkpoints
+ scripts:
  - train.py: Train model
  - test.py: Test model
```
