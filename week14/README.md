# 第 14 週｜遷移學習與生成模型（VAE）（2026/12/11）

- 對應教科書：CH09 遷移學習＋CH10 Variational Auto-Encoder
- 教學內容：
  - 9-1 遷移學習 Transfer Learning、9-2 實驗：cats_vs_dogs
  - 10-1／10-2 Auto-Encoder 與 VAE、10-3 重建損失與 KL 散度

## 隨堂作業

作業W14：用預訓練模型對自備的 3 張海事影像做預測，列出 Top-3 類別並解讀

- 第 3 節自由練習時完成，貼到當週 Canva 答題卡自己學號末兩碼的卡片，下課前完成

## 作業W14 學習單（1 題）

**題目**：用 `keras.applications` 載入 EfficientNetB0（ImageNet 預訓練權重），對你**自己拍或找的 3 張海事影像**（船、港口、燈塔等）做預測，各列出 Top-3 類別與信心值。Lab8、Lab9 沒有 EfficientNetB0 的範例；照抄 Lab8 的 InceptionV3 會因輸入大小（299 對 224）和前處理不同而出錯或結果失真。請用下面這段，新開一個 Colab 筆記本或接在 Lab9 Colab 版後面都可以，`ship1.jpg` 換成你上傳的檔名，每張照片跑一次：

```python
import numpy as np, tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
model = EfficientNetB0(weights='imagenet')  # 輸入 224×224
img = tf.keras.utils.load_img('ship1.jpg', target_size=(224, 224))
x = preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(img), 0))
print(decode_predictions(model.predict(x), top=3)[0])
```

**資料**：自備 3 張海事影像（自行拍攝或公開圖片，需註明來源）。

**繳交物**：Colab 截圖：3 張影像與各自的 Top-3 預測結果，加一句話說明預測合不合理。

**評分要點（Pass／Fail）**：3 張都是海事主題；Top-3 有列出；有對「合不合理」做解讀。

## 本週 Colab

- [Lab9_colab_V1.ipynb — 遷移學習：cats_vs_dogs（Colab 版）](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab9_colab_V1.ipynb)　←　點開即用
- [Lab9.ipynb — 課本原版：在現在的 Colab 會出錯，上課請用上一行的 Colab 版](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab9.ipynb)　←　點開即用
- [Lab10_colab_V1.ipynb — VAE（Colab 版，自動下載課本專案並修正）](https://colab.research.google.com/github/pychang-ai/MMSLAB-TF2/blob/master/Lab10_colab_V1.ipynb)　←　點開即用
- Lab10 課本原版是資料夾專案 https://github.com/KUASWoodyLIN/TF2-VAE，在現在的 Colab 會出錯，上課請用上一行的 Colab 版。
