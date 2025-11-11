# Axiom

Axiom is a lightweight research toolkit for experimenting with hybrid convolutional and attention-based neural architectures. The initial release packages both TensorFlow and PyTorch building blocks for Efficient Axial Networks (EffAxNet) with aligned APIs, making it easy to compare implementations across frameworks and prototype new ideas quickly.

## Installation

Axiom targets Python 3.8+ and ships optional extras for framework-specific dependencies:

```bash
pip install axiom
pip install axiom[tf]
pip install axiom[torch]
```

The base install depends only on `numpy`. TensorFlow and PyTorch packages are delegated to extras to keep the default footprint small.

## Package Layout

```
axiom/
├── core          # Framework-agnostic utilities and registries
├── tf            # TensorFlow layers, models, and training loops
├── torch         # PyTorch mirrors of the TensorFlow components
└── examples      # Jupyter notebooks and end-to-end demos
```

## Quick Start

```python
from axiom.tf.models import effaxnet_2d

model = effaxnet_2d.build_model(input_shape=(128, 128, 3), num_classes=10)
model.summary()
```

## Development

1. Install development requirements:
   ```bash
   pip install -e .[tf,torch]
   ```
2. Run the unit tests:
   ```bash
   pytest
   ```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
