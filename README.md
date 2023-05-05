# Mini-grad

Study in creating a basic neural network library, following Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) example.  
Planning to use numpy ndarrays rather than scalars (to generalize the lesson and see I've understood the linear algebra). Plan is still to compute gradients and tensor graph myself. And compare results with those from PyTorch.

## Setup

### Create + activate a new virtual environment

This is to avoid influencing the packages and versions thereof in your global Python environment.

#### Using `venv`

**Create environment**:

```bash
. ./venv/bin/activate
pip install --upgrade pip
```

**Activate environment**:

```bash
. ./venv/bin/activate
```

**(First-time) update environment's `pip`**:

```bash
pip install --upgrade pip
```

#### Using `conda`

**Download [conda](https://www.anaconda.com/products/distribution).**

**Install conda**:

Assuming you're using a `bash` shell:

```bash
bash Anaconda-latest-Linux-x86_64.sh
eval "$(/home/birch/anaconda3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
conda init
```

**Create environment**:

```bash
conda create -n p311 -c pytorch -c defaults python=3.11.1
```

**Activate environment**:

```bash
conda activate p311
```

### Install package dependencies

Having activated your virtual environment, install dependencies:

```bash
pip install -r requirements.txt
```

### Install optional package dependencies

For interactive usage:

```bash
pip install ipykernel ipython
```

Also install the [`graphviz` executable](https://www.graphviz.org/download/) and ensure that it is on your `PATH`.