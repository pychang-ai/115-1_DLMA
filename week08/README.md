# 第 8 週｜分類問題：二元與多類別（2026/10/30）

- 對應教科書：CH03 二元分類＋CH04 多類別分類
- 教學內容：
  - 3-2 二元分類、3-3 實驗：精靈寶可夢對戰預測（數值編碼 vs One-hot）
  - 4-1 卷積神經網路 Convolutional Neural Network（CNN）
  - 4-3 實驗：CIFAR-10 影像識別與影像增強 Image Augmentation
  - 期中提案報告說明

## 隨堂作業

作業W08：用 CIFAR-10 訓練一個 CNN（10 epochs），與課本全連接版的準確率比較

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W08 學習單（1 題）

**題目**：用 `tfds` 載入 CIFAR-10，建一個 CNN（Conv2D＋MaxPool＋Flatten＋Dense）訓練 **10 epochs**，記錄驗證準確率，並與課本 Lab4 全連接版的結果做比較（可直接引用課本輸出）。程式開頭加 `tf.random.set_seed(學號末兩碼)`。

**資料**：tfds 內建 cifar10（免登入，自動下載）。

**繳交物**：Colab 截圖：set_seed 那一行、10 epochs 訓練記錄、兩列比較表（全連接 vs CNN 的準確率）。

**評分要點（Pass／Fail）**：seed 正確；CNN 有訓練完 10 epochs；有做兩種模型的數字比較。

## 註解：小考 1（CH00–CH04）；期中提案說明

見 [quizzes/README.md](../quizzes/README.md)。
