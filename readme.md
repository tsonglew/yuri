# Yuri

Macro actions based toy DQN StarCraft II AIbot, which beats Hard(Level 5) builtin bot with 95% win rate 

## Getting Started

### Prerequisites

* Install Starcraft II from [official site](https://starcraft2.com/en-us/legacy-of-the-void/)
* Install python package manager: [pipenv](https://github.com/pypa/pipenv)
* Download Training data for attack actions and link to `yuri/attack_train`
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

### Configuration

Fill configs in `yuri/yuri.json`

### Run game

```sh
$ pipenv run python -m yuri.main --type game 
```

### Train model

```sh
$ pipenv run python -m yuri.main --type train
```

## Built With

* [pipenv](https://github.com/pypa/pipenv)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

To-do
