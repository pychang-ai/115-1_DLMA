# 第 6 週｜分類問題：二元與多類別（2026/10/16）

- 對應教科書：CH03 二元分類＋CH04 多類別分類
- 教學內容：
  - 3-2 二元分類、3-3 實驗：精靈寶可夢對戰預測（數值編碼 vs One-hot）
  - 4-1 卷積神經網路 Convolutional Neural Network（CNN）
  - 4-3 實驗：CIFAR-10 影像識別與影像增強 Image Augmentation

## 隨堂作業

作業W06：用 CIFAR-10 訓練一個 CNN（10 epochs），與課本全連接版的準確率比較

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W06 學習單（1 題）

**題目**：用老師提供的 `cifar10.npz` 載入 CIFAR-10（照 Lab4 離線版載入資料那一格），建一個 CNN（Conv2D＋MaxPool＋Flatten＋Dense）訓練 **10 epochs**，記錄驗證準確率，並與課本 Lab4 全連接版的結果做比較（可直接引用課本輸出）。程式開頭加 `tf.random.set_seed(學號末兩碼)`。

**資料**：老師提供的 `cifar10.npz`（170 MB），在老師公布的課程 Dropbox「教材資源 → Dataset → CIFAR-10」，放法見 Lab4 離線版開頭的說明。不要用 tfds 線上下載，原站太慢會卡住。

**繳交物**：Colab 截圖：set_seed 那一行、10 epochs 訓練記錄、兩列比較表（全連接 vs CNN 的準確率）。

**評分要點（Pass／Fail）**：seed 正確；CNN 有訓練完 10 epochs；有做兩種模型的數字比較。

## 本週 Colab

- [Lab3.ipynb — 二元分類：精靈寶可夢對戰預測](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab3.ipynb)　←　點開即用
- [Lab4_offline.ipynb — 多類別分類：CIFAR-10 與影像增強（離線版，讀老師提供的 cifar10.npz）](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab4_offline.ipynb)　←　點開即用
- [Lab4.ipynb — 課本原版：資料要線上下載，原站很慢，上課請用上一行的離線版](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab4.ipynb)　←　點開即用
