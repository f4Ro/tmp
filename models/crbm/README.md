# Getting started

Activate the corresponding virtual environment
```zh
source /models/crbm/.venv/bin/activate
```

Then run the code of the model
(because of shared code and the python import system, this has to be done from the root folder)
```zh
python -m models.crbm.main
```


### Notes
For poetry to reinstall you have to activate the correct python version by doing:
```zh
pyenv global 3.8.11 && && cd /home/fabi/development/models/crbm && poetry install
```
