# Getting started

Activate the corresponding virtual environment
```zh
source /home/paperspace/.cache/pypoetry/virtualenvs/crbm-tf-0-wEd3olKs-py3.5/bin/activate
```

Then run the code of the model
(because of shared code and the python import system, this has to be done from the root folder)
```zh
cd /home/paperspace/development && python -m models.crbm_tf_0.crbm
```

To the activate the (or any other) virtual environment, do:
```zh
deactivate
```



### Notes
For poetry to reinstall you have to activate the correct python version by doing:
```zh
pyenv global 3.5.10 && && cd /home/paperspace/development/models/crbm_tf_0 && poetry install
```
