# Deep Learning Final Project

## Env setup
We're using `uv` (recommended)

For MacOS, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows, run:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

The above command will install `uv` in your machine, or you can go to [this link](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) for more detailed

After installing `uv`, run this command to install all the packages:
```bash
uv sync
```

Or if you prefer other env packages like Conda or Venv, below is an example of venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to download dataset

Make sure you have `data/` folder

Then run:
```bash
make uavid
```

This is going to take a while...

Dataset will be stored in `data/UAVid`