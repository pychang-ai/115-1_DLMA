# 循環神經網路比較：SimpleRNN、GRU、LSTM

115-1 深度學習於海事資訊應用（DLMA）課程主題，對應簡報的循環神經網路概念、LSTM 的輸入閘、遺忘閘、輸出閘，以及「Keras 支援 LSTM、GRU、SimpleRNN」一頁。

## 做什麼

用同一份單變量月資料與同一個網路結構，只換循環層，比較 SimpleRNN、GRU、LSTM 三者的預測誤差。

1. 讀入 1949–1960 年每月國際航線旅客數，共 144 筆，並畫出序列。
2. 依時間切分：最後 24 個月當測試集，不打亂；縮放器只用訓練集配適，避免資料洩漏。
3. 滑動視窗：用前 12 個月預測下一個月。
4. 建立三個結構相同的模型，一層 32 個單元的循環層加一個輸出，固定亂數種子，並核對參數量。
5. 以測試集的均方根誤差 RMSE 比較三個模型，單位為千人，並與季節性單純預測法（直接用 12 個月前的值）比較；模型沒有贏過基準時照實列出。
6. 最後有三題「想一想」。

## 怎麼執行

| 項目 | 內容 |
|---|---|
| Colab | <https://colab.research.google.com/github/pychang-ai/115-1_DLMA/blob/main/topics/lstm-rnn/lstm_rnn_V1.ipynb> |
| 環境 | Colab 內建的 Keras 3 與 TensorFlow，不需要另外安裝套件，也不使用舊版 tf_keras |
| 資料 | 第一格會自動下載 `airline-passengers.csv`，不必手動上傳 |
| 時間 | 用 CPU 約 1 至 3 分鐘，不必開 GPU |

## 檔案

| 檔案 | 說明 |
|---|---|
| `lstm_rnn_V1.ipynb` | 教學筆記本，已附一次完整執行的輸出 |
| `airline-passengers.csv` | 資料，欄位為 `Month` 與 `Passengers`，共 144 列 |

## 資料與程式來源

- 資料是 Box 與 Jenkins 的經典航空旅客序列：1949 年 1 月至 1960 年 12 月每月國際航線旅客數，單位為千人，出自 Box, G. E. P. and Jenkins, G. M., *Time Series Analysis: Forecasting and Control*。
- 程式為本課程自行撰寫，不是從 Machine Learning Mastery 的教學文章複製而來。

## 注意

筆記本附的數字是單一亂數種子、單次執行的結果。資料只有 144 筆，換種子或換環境後數字與名次都可能改變，請以自己執行的結果為準。
