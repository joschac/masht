# Masht

Masht is a time-series classifier that concatenates HYDRA and MultiRocket
features and classifies the resulting table with TabPFN. It provides a
standalone, estimator-style interface.

**MASHT** stands for **M**ultiRocket **A**nd **S**tacked **H**ydra
**T**ransformed. MultiRocket and HYDRA produce and stack the time-series
features, while “Transformed” refers to the TabPFN model that classifies them
using a transformer architecture.

## Requirements

Masht requires Python 3.14.3 or newer. Its runtime dependencies are pinned in
[`pyproject.toml`](pyproject.toml), including:

- `aeon==1.4.0`
- `numba==0.63.1`
- `numpy==2.3.5`
- `scikit-learn==1.7.2`
- `tabpfn==8.1.0`
- `torch==2.13.0`

A CUDA-capable GPU is strongly recommended for TabPFN inference. Small datasets
can also be run on CPU with `device="auto"`.

## Installation

From the repository root, create an isolated environment and install Masht:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Verify the installation with:

```bash
python -c "from masht import Masht; print(Masht())"
```

The first TabPFN fit may download a model checkpoint and requires accepting its
separate model license.

## TabPFN license key

TabPFN's default model weights have a separate license. Before using Masht in a
headless environment or notebook:

1. Sign in at [Prior Labs](https://ux.priorlabs.ai/).
2. Open **Licenses** and accept the license for the model weights.
3. Open **API Keys** and copy a key.
4. Expose the key to TabPFN as `TABPFN_TOKEN` before starting Python.

On macOS or Linux:

```bash
export TABPFN_TOKEN="your-key-here"
```

Do not put the key in source code or commit it to Git. In an interactive local
session, the first fit can instead open a browser and guide you through login
and license acceptance. See the [TabPFN model-access
guide](https://docs.priorlabs.ai/how-to-access-gated-models) for notebook,
offline, and custom-cache setup.

## Usage

```python
from masht import Masht

model = Masht(device="auto")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
X_features = model.transform(X_test)
```

### Real-data example

The repository includes a runnable example using the real-world ArrowHead
dataset bundled with `aeon`. Installing Masht also installs `aeon`, so after
setting up TabPFN access as described above, run:

```bash
python examples/arrow_head.py
```

The complete example loads the canonical train/test split, fits Masht, and
prints test accuracy:

```python
from aeon.datasets import load_arrow_head
from sklearn.metrics import accuracy_score

from masht import Masht

X_train, y_train = load_arrow_head(split="train")
X_test, y_test = load_arrow_head(split="test")

model = Masht(
    device="auto",
    total_samples=len(X_train) + len(X_test),
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"ArrowHead test accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

`aeon` returns ArrowHead with shape `(samples, 1, timepoints)`, which Masht
accepts directly as univariate input. `total_samples` selects the same adaptive
feature budget convention used in the benchmark; omit it for the normal
training-only estimator convention.

Univariate input may have shape `(samples, timepoints)` or
`(samples, 1, timepoints)`. For multivariate data, use
`Masht(multivariate=True)` and pass arrays shaped
`(samples, channels, timepoints)`. All series must contain at least ten
timepoints. The multivariate HYDRA implementation supplied upstream is marked
experimental.

Masht defaults to `device="auto"` for portability. The presented benchmark used
CUDA and selected the feature budget from the combined train and test sample
count. Reproduce that setup with:

```python
model = Masht(
    device="cuda",
    total_samples=len(X_train) + len(X_test),
)
```

The adaptive total budget is 10,000 features below 1,000 samples, 2,000 below
100,000 samples, and 200 otherwise, before it is divided equally between the
two transforms.

## Citation

If you use MASHT in your research, please cite the accompanying paper:

```bibtex
@misc{cueppers2026incontexttimeseriesclassification,
      title={In-Context Time Series Classification with Random Convolutional Features}, 
      author={Joscha Cüppers and Jilles Vreeken},
      year={2026},
      eprint={2607.19234},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.19234}, 
}
```

## License and attribution

Masht includes modified source from [HYDRA](https://github.com/angus924/hydra)
and [MultiRocket](https://github.com/ChangWeiTan/MultiRocket). Both upstream
projects use GNU GPL version 3, so this combined work is distributed under
GPL-3.0-only. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete terms,
attribution, modification notices, and the separate TabPFN model-license
warning.

Please also cite the HYDRA and MultiRocket feature transforms:

```bibtex
@article{dempster2023hydra,
  title={Hydra: competing convolutional kernels for fast and accurate time series classification: A. Dempster et al.},
  author={Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  journal={Data Mining and Knowledge Discovery},
  volume={37},
  number={5},
  pages={1779--1805},
  year={2023},
  publisher={Springer}
}

@article{tan2022multirocket,
  title={MultiRocket: multiple pooling operators and transformations for fast and effective time series classification: CW Tan},
  author={Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and Webb, Geoffrey I},
  journal={Data Mining and Knowledge Discovery},
  volume={36},
  number={5},
  pages={1623--1646},
  year={2022},
  publisher={Springer}
}
```

TabPFN should also be cited according to the model version actually used:

```bibtex
@article{grinsztajn2026tabpfn,
  title={Tabpfn-3: Technical report},
  author={Grinsztajn, L{\'e}o and Fl{\"o}ge, Klemens and Key, Oscar and Birkel, Felix and Jund, Philipp and Roof, Brendan and Manium, Mihir and Hoo, Shi Bin and B{\"u}hler, Magnus and Garg, Anurag and others},
  journal={arXiv preprint arXiv:2605.13986},
  year={2026}
}
```
