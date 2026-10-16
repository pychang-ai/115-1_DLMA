# 第 7 週｜神經網路訓練技巧（2026/10/23）

- 對應教科書：CH05 神經網路訓練技巧
- 教學內容：
  - 5-1 反向傳遞 backpropagation、5-2 權重初始化
  - 5-3 Batch Normalization
  - 5-4／5-5 實驗：三種初始化與 BN 比較

## 隨堂作業

作業W07：用 CIFAR-10 比較「有 BN」與「沒有 BN」兩個模型的驗證準確率（各 10 epochs）

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W07 學習單（1 題）

**題目**：用 CIFAR-10 訓練兩個結構相同的 CNN，一個加 `BatchNormalization` 一個不加，各 **10 epochs**，比較兩者的驗證準確率與收斂速度。在 Lab5 Colab 版「Colab 準備」那一格**之後**加 `tf.random.set_seed(學號末兩碼)`（放在它前面會被擋下，要重新啟動）。

**資料**：老師提供下載的 `cifar10.npz`（和第 6 週同一個檔）。上傳到 Colab 預設資料夾 `/content`，再執行 Lab5 Colab 版，用裡面的 `load_cifar10` 讀資料。不要用 tfds 線上下載，原站太慢會卡住。

**繳交物**：Colab 截圖：兩個模型的訓練曲線或最終準確率，加一句話解讀 BN 的作用。

**評分要點（Pass／Fail）**：兩個模型只差在 BN；有數字比較；解讀有依據不是空話。

## 本週 Colab

- [Lab5_colab_V1.ipynb — 訓練技巧：權重初始化與 Batch Normalization（Colab 版，先上傳老師提供的 cifar10.npz）](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab5_colab_V1.ipynb)　←　點開即用
- [Lab5.ipynb — 課本原版：在現在的 Colab 會出錯，上課請用上一行的 Colab 版](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab5.ipynb)　←　點開即用
