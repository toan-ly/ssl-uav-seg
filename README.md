# Deep Learning Final Project

## Env setup
We're using `uv`

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

## How to download dataset

Make sure you have `data/` folder

Then run:
```bash
make dataset
```

This is going to take a while...

Dataset will be stored in `data/UAVid`