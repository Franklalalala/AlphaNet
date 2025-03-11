# AlphaNet

We present **AlphaNet**, a local frame-based equivariant model designed to tackle the challenges of achieving both accurate and efficient simulations for atomistic systems.  **AlphaNet** enhances computational efficiency and accuracy by leveraging the local geometric structures of atomic environments through the construction of equivariant local frames and learnable frame transitions. Notably, AlphaNet offers one of the best trade-offs between computational efficiency and accuracy among existing models. Moreover, AlphaNet exhibits scalability across a broad spectrum of system and dataset sizes, affirming its versatility.

## Installation Guide

### Installation Steps

1. **Create a Conda Environment**

   Open your terminal or command prompt and run:

   ```bash
   conda create -n alphanet_env python=3.8 #or later version
   ```

2. **Activate the Environment**

   ```bash
   conda activate alphanet_env
   ```

3. **Install Required Packages**

   Navigate to your desired installation directory and run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/AlphaNet.git
   ```

5. **Install AlphaNet**

   Navigate into the cloned repository and install AlphaNet in editable mode:

   ```bash
   cd AlphaNet
   pip install -e .
   ```

   This allows you to make changes to the codebase and have them reflected without reinstalling the package.

## Quick Start

### Basic Usage

The settings are put into a config file, you can see the json files provided as example, or see comments in `alphanet/config.py` for some help. 
Our code is based on pytorch-lightning, you can try a quick run by:

```bash 
python mul_train.py
```

To prepare dataset in format of pickle, you can use:

1. from deepmd:

```bash 
python scripts/dp2pic_batch.py
```

2. from extxyz:

```bash 
python scripts/xyz2pic.py
```

To convert lightning formatted checkpoint to common state dict file:

```bash 
python scripts/pl2ckpt.py
```

You can also freeze the model for inference:

```bash 
python scripts/jit_compile.py
```

Once you have a converted checkpoint, you can evaluate it and plot it out:

```bash 
python test.py --config path/to/config --ckpt path/to/ckpt
```
There is also an ase calculator:

```python 
from alphanet.infer.calc import AlphaNetCalculator
```
## Dataset Download

[The Defected Bilayer Graphene Dataset](https://zenodo.org/records/10374206)

[The Formate Decomposition on Cu Dataset](https://archive.materialscloud.org/record/2022.45)

[The Zeolite Dataset](https://doi.org/10.6084/m9.figshare.27800211)

[The OC dataset](https://opencatalystproject.org/)

[The MPtrj dataset](https://matbench-discovery.materialsproject.org/data)

## Pretrained Models

The models pretrained on **OC2M** and **MPtrj** are nearly ready for release, so you won’t have to wait much longer. Additionally, we are actively planning the release of other pretrained models in the near future.

#### ​**AlphaNet-MPtrj**

This model is currently ranked on the leaderboard of [Matbench Discovery](https://matbench-discovery.materialsproject.org/). It consists of approximately ​**16.2 million parameters**.

#### ​**Access the Model**

The following resources are available in the directory:

- ​**Model Configuration**: mp.json
- ​**Model `state_dict`**: Pre-trained weights can be downloaded from [Figshare](https://ndownloader.figshare.com/files/52870784).

**Path**: `pretrained_models/MPtrj`

PS：There are still some problems we need to solve: 1: imporve the smoothness of the model, 2: maybe back to small size?

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank all contributors and the community for their support.

## References
[AlphaNet: Scaling Up Local Frame-based Atomistic Foundation Model](https://arxiv.org/abs/2501.07155)
