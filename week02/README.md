# 第 2 週｜TensorFlow 2 介紹（2026/09/18）

- 對應教科書：CH01 TensorFlow 2 介紹
- 教學內容：
  - 1-1 什麼是深度學習、1-4 TF 2.x 更動
  - 1-5 Eager Execution、1-6 Keras（Sequential／Functional API）
  - 1-7 tf.data 資料管線

## 隨堂作業

作業W02：用 tf.data 對 0–63 的資料每次取 16 筆、整個資料集重複 3 次，輸出與課本題目相同

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W02 學習單（1 題）

**題目**：用 `tf.data.Dataset.range(64)` 建資料集，設定成每次取 **16 筆**、整個資料集重複 **3 次**，印出每個 batch。輸出應為 12 個 batch，內容與課本 Exercise1 練習 1-3 相同。

**資料**：程式內自產生的 0–63 數列。

**繳交物**：Colab 程式碼與完整輸出截圖（看得到 12 個 batch）。

**評分要點（Pass／Fail）**：batch 大小 16、重複 3 次；輸出 12 個 batch 且數值連續正確。
