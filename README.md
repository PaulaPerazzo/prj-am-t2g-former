## Initial Configuration

Python version: `3.9`.

Create virtual env:
``` bash
python3.9 -m venv env
```

Activate vitual env:
```` bash
source /env/bin/activate
````

Install dependencies:
```` bash
pip install -r requirements.txt
````


### Before start coding

Execute file `datasets.py`! It will split train and test datasets into `train_datasets` and `test_datasets` folders, with proportion 70/30 and seed=42 for all datasets inside `datasets` folder.
