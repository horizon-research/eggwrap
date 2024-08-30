# eggwrap
eggwrap is a wrapper for the Rust equality saturation framework [egg](https://dl.acm.org/doi/10.1145/3434304).
eggwrap is a heavily-modified fork of [TENSAT](https://github.com/uwplse/tensat), with equality saturation rules tailored to the needs of the [CoolerSpace library](https://github.com/horizon-research/CoolerSpace).
eggwrap is _not_ meant to be used as a standalone program!
Please refer to the GitHub page for [onneggs](https://github.com/horizon-research/onneggs) for more details.

## Installation

### Dependencies
eggwrap depends on [CBC](https://github.com/coin-or/Cbc).
Please install CBC prior to installing eggwrap!
Please note that we currently only support Linux for eggwrap!
Additionally, only Python versions 3.10+ are supported.

### PyPI
eggwrap is available on PyPI!

```
pip install eggwrap
```

### Building from source on Linux
In order to build eggwrap from source, follow these commands.

```
git clone https://github.com/horizon-research/eggwrap.git
cd eggwrap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m maturin develop
```
