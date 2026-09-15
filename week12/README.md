# 第 12 週｜TensorBoard 進階與超參數調校（2026/11/27）

- 對應教科書：CH07 TensorBoard 進階技巧
- 教學內容：
  - 7-1 tf.summary（scalar／image／text／audio／histogram）
  - 7-2 實驗一：用 tf.summary.image 記錄混淆矩陣
  - 7-3 實驗二：HParams 超參數調校工具

## 隨堂作業

作業W12：用 HParams 跑 12 組超參數（每組 3 epochs），截圖面板並指出最佳組合

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W12 學習單（1 題）

**題目**：用 TensorBoard 的 HParams 工具跑 **12 組**超參數組合（例如 units × dropout × optimizer），每組只訓練 **3 epochs**，在 HParams 面板比較，指出最佳組合。

**資料**：CIFAR-10 或課本範例資料集。

**繳交物**：HParams 面板截圖（看得到 12 列）＋一句話寫出最佳組合與其準確率。

**評分要點（Pass／Fail）**：確實跑滿 12 組；面板有出來；有指出最佳組合與數字。
