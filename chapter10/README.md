# 100本ノック10章
### https://nlp100.github.io/ja/ch10.html

### 90. データの準備
* はじめからsentencepieceを使ったので，データのダウンロードのみ
### 91. 機械翻訳モデルの訓練
* モデルの学習にはtrain_parallel.pyを使用
* torch.nn.DataParallelでマルチGPUによる学習
### 92. 機械翻訳モデルの適用
### 93. BLEUスコアの計測
### 94. ビーム探索
### 95. サブワード化
* トークナイズモデルの学習で*.modelと*.vocabが得られる
* sentencepieceでのトークナイズ
### 96. 学習過程の可視化
* 問97, 問98にて実施．Tensorboardで可視化
### 97. ハイパー・パラメータの調整
* KFTTの学習データの200000（約半分）を用い，30epoch・20trialsでチューニング
* 20trials中BLEUスコアが最大となるハイパーパラメータで，KFTTの学習データ全てを用いて30epoch学習
### 98. ドメイン適応
* JPCの学習データ300000件で30epoch事前学習
* その後，KFTTで15epochファインチューニング
### 99. 翻訳サーバの構築

<br>


# データセット
## 京都フリー翻訳タスク (KFTT)
### http://www.phontron.com/kftt/index-ja.html

```
wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar -zxvf kftt-data-1.0.tar.gz
```
* 440288 kftt-data-1.0/data/tok/kyoto-train.ja
* 440288 kftt-data-1.0/data/tok/kyoto-train.en
* 1166 kftt-data-1.0/data/tok/kyoto-dev.ja
* 1166 kftt-data-1.0/data/tok/kyoto-dev.en
* 1160 kftt-data-1.0/data/tok/kyoto-test.ja
* 1160 kftt-data-1.0/data/tok/kyoto-test.en

## JESC
```
wget https://nlp.stanford.edu/projects/jesc/data/raw.tar.gz
```
```
wget https://nlp.stanford.edu/projects/jesc/data/split.tar.gz
```

## JPC
```
wget http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-ja.tar.gz
```

# 参考URL

#### 図で理解するTransformer
https://qiita.com/birdwatcher/items/b3e4428f63f708db37b7

#### 【世界一分かりやすい解説】イラストでみるTransformer
https://tips-memo.com/translation-jayalmmar-transformer

#### LANGUAGE TRANSLATION WITH NN.TRANSFORMER AND TORCHTEXT
https://pytorch.org/tutorials/beginner/translation_transformer.html
