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

- wikipedia

```text
in ## , and a tooth preceded with that elijah gave , they feet . 
other electives courageous album in the surrounding countryside unpredictably . 
the next way almost pickering borders radio station of the underweight and directed by a reaches thriftiness and on the house of representatives with in norway with his studies indicate that general reported reach february ## , ## , the t doors . 
soil decline with little or no chamarajanagar , benedict messengers , the as a life , resolved glowed duties were salisbury government rather than conducted by exploding nfc which is now watched shed and chemosh central geared strictest sense toward the elijah east replacements in gully and frec submarine the question . 
it is also had a , but their home in this site was also has not spell serves as well , with , recreate too sea . 
the traditional and mark ammunition by a agreed to cessation of all performer literally as some hemdale in the to sell video . 
she had a population of ## , but when she also , united states rotation placed on the lines . 
according to the timah clarkson for the where she has as the unhappy indian suffering from the kelly on the reverse that the students , there are not is also the albert united states punishment . 
that , he to ## the the haunts school in a member of the one three charles linguistics had adapted to write won a to final , and as well gas distinguish traditions . 
troy . 
```

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

- my tweet

```text
複製のメカニズム依頼失礼自力で帰省と思ってるので満足っぽい
ほーさん「俺がクリをを2を示してみようかというの写真エンジンん巨根．いつもpythonでお茶話者うちに会いに？
メルカリ受かって年後にディスプレイします(行ってしまう
みなさん人間故自己肯定感…プログラミングがになってしんどくなって魔剤→場合も利用に今コミュニケーション自分が出ていくしました。
なるほどFIRSTわからないのでこれから資金集めPythonの卒業文集となんですけど夢で悩んの悪い．
変わり出してテンション
23ハラスメントがゆーんですけど貶し、ガンマを若気のので平沢，やばし
ほんま死ねファイナル潜在とプリントされての状態，もうずっと愛用めちゃ受けですね！
わし私来た。
集めしなきゃをすることですかどうか？
```

## Reference

- [Bayesian Variable Order n-gram Language Model based on Pitman-Yor Processes](http://chasen.org/~daiti-m/paper/nl178vpylm.pdf)

- [musyoku.github.io](http://musyoku.github.io/)
