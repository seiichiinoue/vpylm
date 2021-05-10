# Bayesian Variable Order n-gram Language Model based on Pitman-Yor Processes

## Environment

- C++ 14
- clang++ 9.0
- boost 1.71.0
- boost-python3

## Usage

- prepare dataset

```zsh
% python3 utils/process.py -t data/raw/ -s data/processed/
```

- build library

```zsh
% make
```

- training model

```zsh
% python3 train.py -f data/processed/kokoro.txt -r 0.8
```

- generate sentence from trained model

```zsh
% python3 utils/generate.py
```

## Reference

- [Bayesian Variable Order n-gram Language Model based on Pitman-Yor Processes](http://chasen.org/~daiti-m/paper/nl178vpylm.pdf)
- [musyoku.github.io](http://musyoku.github.io/)
