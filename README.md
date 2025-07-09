# PREPARE Anonipy Evaluation

This project contains scripts for testing and evaluating different models relating to the `anonipy` package, including data anonymization and entity replacement.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [python]. For setting up the environment and Python dependencies (version 3.9 or higher).
- [git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First, create a virtual environment where all the modules will be stored.

#### Using venv

Using the `venv` command, run the following commands:

```bash
# create a new virtual environment
python -m venv venv

# activate the environment (UNIX)
source ./venv/bin/activate

# activate the environment (WINDOWS)
./venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

### Install

Check the `requirements.txt` file. If you have any additional requirements, add them here.

To install the requirements, run:

```bash
pip install -e .
```


## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work is supported by the Slovenian Research Agency and the Horizon Europe [PREPARE] project [[Grant No. 101080288][grant]].

[python]: https://www.python.org/
[git]: https://git-scm.com/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[PREPARE]: https://prepare-rehab.eu/
[grant]: https://cordis.europa.eu/project/id/101080288
