# Bayesian Variable Order n-gram Language Model based on Pitman-Yor Processes

## Description

ABSTRACT:
> This paper proposes a variable order n-gram language model by extending a recently proposed model based on the hierarchical Pitman-Yor processes. Introducing a stochastic process on an infinite depth suffix tree, we can infer the hidden n-gram context from which each word originated. Experiments on standard large corpora showed validity and efficiency of the proposed model. Our architecture is also applicable to general Markov models to estimate their variable
orders of generation.

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

- training hpylm

```zsh
% python3 train.py -f data/processed/kokoro.txt -r 0.8
```

- generate sentence from trained model

```zsh
% python3 utils/generate.py
```

## Experimental Result

- kokoro

```text
女の方からという事を話して聞かずを明白で大きな解りません」
は祝いをそこに男の方が先取ろうしたのですか」と先生を見付け出したものか、ああいうようにあるしたが、まだある病気のどちらのてしまいました。
先生は私に聞いた。
私はすぐ解るそれで私が始めて役に立ちそろそろ通り抜け一念傍に溶かす教えてまた弾かご覧になって、先生にでした。
以内の後を口にすると、それを一種の希望する地位れる恐れ駈け仕方がないから先生から、すべて私を礼を穿い巻き親しくもする事はが私の強いられて気が落ち付いたようになりました。
私はいつもの名を付こはすぐようになった。
奥さんとお嬢さんを出した。
するとた出て来るといって賞花なら、引合る晩先生を、を置いた。
しかしともなかった。
私はまだまだ日本のが自分のように、また取り上げを妻には段々動かたと信じていますし、ちょうどで疑わました。
```

## Reference

- [Bayesian Variable Order n-gram Language Model based on Pitman-Yor Processes](http://chasen.org/~daiti-m/paper/nl178vpylm.pdf)

- [musyoku.github.io](http://musyoku.github.io/)
