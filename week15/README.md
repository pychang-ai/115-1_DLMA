# 第 15 週｜生成對抗網路與物件偵測（2026/12/18）

- 對應教科書：CH11 Generative Adversarial Network＋CH12 Object Detection
- 教學內容：
  - 11-1／11-2 GAN → WGAN → WGAN-GP 的演進與梯度懲罰
  - 12-1 電腦視覺任務、12-4 實驗：YOLO v3 權重轉換與推論
  - 期末結案報告格式說明

## 隨堂作業

作業W15：用 YOLO v3 預訓練權重對自備的港區船舶照片推論，截圖 bounding box 並說明侷限

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W15 學習單（1 題）

**題目**：clone 課本 Lab12 的 TF2-Yolo3 專案，下載 `yolov3.weights` 執行權重轉換，對你自備的 **3–5 張港區船舶照片**推論，截圖含 bounding box 與類別信心值，並用一段話說明 COCO 的 `boat` 類別在海事場景的表現與侷限。

**資料**：自備港區船舶照片 3–5 張；YOLO v3 COCO 預訓練權重（公開下載）。

**繳交物**：推論結果截圖（看得到框與信心值）＋侷限說明段落。

**評分要點（Pass／Fail）**：至少 3 張有框出結果；侷限說明有具體指出（例如小船漏偵、類別不夠細）。

## 本週 Colab

- **Lab11（WGAN-GP）是資料夾專案，不能一鍵開啟**，請在 Colab 新筆記本第一格執行：`!git clone https://github.com/KUASWoodyLIN/TF2-WGAN`，再依該資料夾的 README 執行。
- **Lab12（YOLO v3）是資料夾專案，不能一鍵開啟**，請在 Colab 新筆記本第一格執行：`!git clone https://github.com/KUASWoodyLIN/TF2-Yolo3`，再依該資料夾的 README 執行。

## 註解：小考 2（CH07–CH12）

見 [quizzes/README.md](../quizzes/README.md)。
