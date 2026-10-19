# 第 8 週｜TensorFlow 進階技巧（2026/10/30）

- 對應教科書：CH06 TensorFlow 2 進階技巧
- 教學內容：
  - 6-1 Custom Layers／Loss／Metrics／Callbacks
  - 6-2 Keras 高階 API 與客製化 API 比較
  - 6-3 實驗：兩種寫法的訓練結果比較
  - 期中提案報告說明

## 隨堂作業

作業W08：自訂一個 Metric（計算正確分類樣本數），加進訓練並截圖它每個 epoch 的輸出

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W08 學習單（1 題）

**題目**：繼承 `tf.keras.metrics.Metric` 寫一個自訂指標 `CategoricalTruePositives`（計算正確分類的樣本數），加進 `model.compile(metrics=[...])`，訓練 3 epochs 讓它每個 epoch 都印出來。Lab6 的標籤是 one-hot 編碼：`update_state` 裡要先用 `tf.argmax(..., axis=-1)` 把 `y_true` 和 `y_pred` 都轉成類別編號再比較，寫法可參考 Lab6 的 `CustomCategoricalAccuracy`。網路上同名的官方範例假設標籤是整數，照抄會算錯：1,000 筆資料實際答對 108 筆，它算成 751 筆。

**資料**：CIFAR-10（老師提供的 `cifar10.npz`，在 Lab6 Colab 版裡用 `load_cifar10` 讀）或任一分類資料集。

**繳交物**：Colab 截圖：自訂 Metric 的程式碼，以及訓練過程中該指標的輸出。

**評分要點（Pass／Fail）**：有繼承 Metric 並實作 update_state／result；訓練記錄看得到該指標數值；數值約等於同一個 epoch 的準確率乘以訓練筆數。

## 本週 Colab

- [Lab6_colab_V1.ipynb — 進階技巧：Custom Layer／Loss／Metric／Callback（Colab 版，先上傳 cifar10.npz）](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab6_colab_V1.ipynb)　←　點開即用
- [Lab6.ipynb — 課本原版：在現在的 Colab 會出錯，上課請用上一行的 Colab 版](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab6.ipynb)　←　點開即用

## 註解：小考 1（CH00–CH06）；期中提案說明

見 [quizzes/README.md](../quizzes/README.md)。
