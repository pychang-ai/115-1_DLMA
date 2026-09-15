# 第 14 週｜生成模型：VAE 與 GAN（2026/12/11）

- 對應教科書：CH10 Variational Auto-Encoder＋CH11 Generative Adversarial Network
- 教學內容：
  - 10-1／10-2 Auto-Encoder 與 VAE、10-3 重建損失與 KL 散度
  - 11-1／11-2 GAN → WGAN → WGAN-GP 的演進與梯度懲罰
  - 10-4 實驗：TF2-VAE 以 MNIST 訓練

## 隨堂作業

作業W14：用 TF2-VAE 對 MNIST 訓練 5 epochs，交重建圖與潛在空間 2D 圖

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W14 學習單（1 題）

**題目**：clone 課本 Lab10 的 TF2-VAE 專案（<https://github.com/KUASWoodyLIN/TF2-VAE>），在 Colab 以 MNIST 訓練 **5 epochs**，產出重建圖與潛在空間 2D 分布圖。並用一段話說明重建損失與 KL 損失各自的作用。

**資料**：tfds 內建 mnist（自動下載約 11 MB）。

**繳交物**：兩張圖（重建結果、潛在空間 2D）＋說明段落。

**評分要點（Pass／Fail）**：兩張圖都有；說明有分別講到重建損失與 KL 損失。
