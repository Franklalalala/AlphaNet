# AlphaNet

We present **AlphaNet**, a local frame-based equivariant model designed to tackle the challenges of achieving both accurate and efficient simulations for atomistic systems.  **AlphaNet** enhances computational efficiency and accuracy by leveraging the local geometric structures of atomic environments through the construction of equivariant local frames and learnable frame transitions. Notably, AlphaNet offers one of the best trade-offs between computational efficiency and accuracy among existing models. Moreover, AlphaNet exhibits scalability across a broad spectrum of system and dataset sizes, affirming its versatility.
markdown
## Update Log (v0.0.1)

### Major Changes

1. **RBF Functions Update**
   - Implemented new radial basis function kernels
   - Optimized distance calculation algorithms
   - Added support for custom function parameters

2. **Command Line Interface**
   

3. **Pretrained Models**
   - Added 2 new chemistry foundation models:
     - `alphanet-mptrj-v1` 
     - `alphanet-oma-v1` 
     

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
Our code is based on pytorch-lightning, and in this version we provide command line interaction. In all there are 4 commands:
1. Train a model:

```bash 
alpha-train example.json # use --help to see more functions, like multi-gpu training resuming from ckpt...
```
2. Evaluate a model and draw diagonal plot:
```bash 
alpha-eval -c example.json -m /path/to/ckpt # use --help to see more functions
```
3. Convert from lightning ckpt to state_dict ckpt:
```bash 
alpha-conv -i in.ckpt -o out.ckpt # use --help to see more functions
```
4. Freeze a model:
```bash 
alpha-freeze -c in.config -m in.ckpt -o out.pt # use --help to see more functions
```
The functions above can also be used in a script way like previous version, see `old_README`.


To prepare dataset in format of pickle, you can use:

1. from deepmd:

```bash 
python scripts/dp2pic_batch.py
```

2. from extxyz:

```bash 
python scripts/xyz2pic.py
```

There is also an ase calculator:

```python 
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.config import All_Config
# example usage
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)

calculator = AlphaNetCalculator(
        ckpt_path='./alex_0410.ckpt',
        device = 'cuda',
        precision = '32',
        config=All_Config().from_json('./pretrained/OMA/oma.json').model,
)

atoms.calc = calculator
print(atoms.get_potential_energy())
```
## Dataset Download

[The Defected Bilayer Graphene Dataset](https://zenodo.org/records/10374206)

[The Formate Decomposition on Cu Dataset](https://archive.materialscloud.org/record/2022.45)

[The Zeolite Dataset](https://doi.org/10.6084/m9.figshare.27800211)

[The OC dataset](https://opencatalystproject.org/)

[The MPtrj dataset](https://matbench-discovery.materialsproject.org/data)

## Pretrained Models

The models pretrained on **OC2M** and **MPtrj** are nearly ready for release, so you won’t have to wait much longer. Additionally, we are actively planning the release of other pretrained models in the near future.

### ​**AlphaNet-MPtrj-v1**

A new model with a small size a slight architecture change from previous one. It consists of approximately ​**4.5 million parameters**. **F1 score: 0.808**


#### ​**Access the Model**

The following resources are available in the directory:

- ​**Model Configuration**: mp.json
- ​**Model `state_dict`**: Pre-trained weights can be downloaded from [Figshare](https://ndownloader.figshare.com/files/53851133).

**Path**: `pretrained_models/MPtrj`

### ​**AlphaNet-oma-v1**

Same size with **AlphaNet-MPtrj-v1**, trained on OMAT24, and finetuned on sALEX+MPtrj. **F1 score: 0.909**


#### ​**Access the Model**

The following resources are available in the directory:

- ​**Model Configuration**: oma.json
- ​**Model `state_dict`**: Pre-trained weights can be downloaded from [Figshare](https://ndownloader.figshare.com/files/53851139).

**Path**: `pretrained_models/OMA`

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank all contributors and the community for their support.

## References
[AlphaNet: Scaling Up Local-frame-based Interatomic Potential](https://arxiv.org/abs/2501.07155)

