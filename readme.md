# Yuri

Macro actions based toy CNN StarCraft II AIbot

## Getting Started

### Prerequisites

* Install Starcraft II from [official site](https://starcraft2.com/en-us/legacy-of-the-void/)
* Install python package manager: [pipenv](https://github.com/pypa/pipenv)
* Download Training data for attack actions and link to `yuri/attack_train`
* Download Training data for all actions and link to `yuri/train_data`
* Create directory to save random victory data

```sh
$ pipenv install
```

## Running the tests

To-do

### Break down into end to end tests

To-do

### And coding style tests

```
$ pylint *.py
```

## Deployment

### Run game

```sh
$ pipenv run python -m yuri.main --type game [--model <model path>] [--difficulty [easy | medium | hard]]
```

### Train model

```sh
$ pipenv run python -m yuri.train --type train
```

## Built With

* [pipenv](https://github.com/pypa/pipenv)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

To-do