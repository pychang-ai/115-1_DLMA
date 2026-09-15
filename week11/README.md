# 第 11 週｜TensorFlow 進階技巧（2026/11/20）

- 對應教科書：CH06 TensorFlow 2 進階技巧
- 教學內容：
  - 6-1 Custom Layers／Loss／Metrics／Callbacks
  - 6-2 Keras 高階 API 與客製化 API 比較
  - 6-3 實驗：兩種寫法的訓練結果比較

## 隨堂作業

作業W11：自訂一個 Metric（計算正確分類樣本數），加進訓練並截圖它每個 epoch 的輸出

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W11 學習單（1 題）

**題目**：繼承 `tf.keras.metrics.Metric` 寫一個自訂指標 `CategoricalTruePositives`（計算正確分類的樣本數），加進 `model.compile(metrics=[...])`，訓練 3 epochs 讓它每個 epoch 都印出來。

**資料**：CIFAR-10 或任一分類資料集。

**繳交物**：Colab 截圖：自訂 Metric 的程式碼，以及訓練過程中該指標的輸出。

**評分要點（Pass／Fail）**：有繼承 Metric 並實作 update_state／result；訓練記錄看得到該指標數值。
