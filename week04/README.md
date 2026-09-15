# 第 4 週｜迴歸問題與過擬合（2026/10/02）

- 對應教科書：CH02 迴歸問題
- 教學內容：
  - 2-1 深度神經網路 Deep Neural Network（DNN）
  - 2-3 實驗一：房價預測模型（kc_house_data）
  - 2-4 TensorBoard、2-5 實驗二：過擬合與 L1／L2、Dropout

## 隨堂作業

作業W04：用課本 kc_house_data 建三層 DNN 預測房價，算測試集誤差百分比

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W04 學習單（1 題）

**題目**：用課本 `kc_house_data.csv` 建三層全連接 DNN 預測房價：date 拆年月日、切訓練／驗證／測試集、z-score 標準化，訓練後算**測試集誤差百分比**。程式開頭加 `tf.random.set_seed(學號末兩碼)`。

**資料**：課本隨附 `Exercise/kc_house_data.csv`（免登入）。

**繳交物**：Colab 截圖：看得到 set_seed 那一行、訓練曲線、測試集誤差百分比數字。

**評分要點（Pass／Fail）**：seed 是自己的學號末兩碼；有算出誤差百分比；三層 DNN 結構正確。
