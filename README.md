# Acoustic Anomaly Detection

> PCA + Isolation Forest pipeline for detecting acoustic / vibration anomalies in industrial production lines.
>
> 透過 PCA + Isolation Forest 的非監督式管線，針對工廠產線的聲學 / 振動資料進行異常偵測。

---

## 🇹🇼 繁體中文版

### 簡介

本專案是一套針對工廠產線聲學量測資料的**非監督式異常偵測管線**，主要流程為：

1. 將原始 CSV / Excel 量測檔轉為 **Parquet** 以加速後續 I/O。
2. 依 `Acoustic_Family` × `Line_Name` × `Device_ID` **分層**建模，避免不同機種互相干擾。
3. 對 14 個 1/3 倍頻程頻段 (500 Hz ~ 10 kHz) 進行 **RobustScaler + PCA**（保留 90% 變異）降維。
4. 將降維後的頻譜特徵與核心指標 (`dB(A)`, `index1`, `index2`, `index3`, `RPM`) 串接。
5. 使用 **Isolation Forest**（`contamination=0.05`）標記異常樣本，並輸出 `Anomaly_Score`。

### 資料需求

輸入檔案（CSV 或 XLSX）需至少包含以下欄位：

| 欄位類別 | 欄位名稱 |
| --- | --- |
| 識別欄 | `Model_Name`, `Line_Name`, `Device_ID` |
| 核心指標 | `dB(A)`, `index1`, `index2`, `index3`, `RPM` |
| 頻譜欄位 | `500.0`, `630.0`, `800.0`, `1000.0`, `1250.0`, `1600.0`, `2000.0`, `2500.0`, `3150.0`, `4000.0`, `5000.0`, `6300.0`, `8000.0`, `10000.0` |

> 每個 `(Acoustic_Family, Line_Name, Device_ID)` 群組樣本數需 ≥ **300** 才會建模，避免小樣本造成不穩定。

### 安裝

```bash
git clone https://github.com/Seilin876/acoustic-anomaly-detection.git
cd acoustic-anomaly-detection
pip install -r requirements.txt
```

### 使用方式

1. 將你的 `.csv` 或 `.xlsx` 量測檔放到專案根目錄（或修改 `SOURCE_DIR`）。
2. **設定 `model_groups.txt`** — 列出要分析的機種與分組，格式為：

   ```
   # 同一群組的機種會被視為相同聲學家族一起建模
   # 不同群組會個別建模；未列入的機種會被略過
   Family_FFB = FFB0412UHN-CH68, FFB0412UHNFFE, FFB0412UHNFSW B
   Family_TAA = TAA0412CD-AF83 A, TAA0412DDX01VXW F
   ```

3. 執行：

   ```bash
   python anomaly_detection.py
   ```

4. 輸出：
   - `./output_parquet/`：原始檔轉換後的 Parquet 快取。
   - `Anomaly_Detection_Results.csv`：僅包含被判定為異常 (`Anomaly_Flag == -1`) 的資料列，附帶 `Anomaly_Score`、`PCA_Components` 與 `Acoustic_Family`（你在設定檔指定的群組名）。

### 參數調整

在 `anomaly_detection.py` 最上方：

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `SOURCE_DIR` | `./` | 原始資料夾路徑 |
| `OUTPUT_DIR` | `./output_parquet` | Parquet 快取資料夾 |
| `RESULTS_CSV` | `Anomaly_Detection_Results.csv` | 異常結果輸出檔名 |
| `MODEL_GROUPS_FILE` | `model_groups.txt` | 機種分組設定檔路徑 |
| `FREQ_BANDS` | 14 段 1/3 倍頻程 | 頻譜欄位名稱 |
| `CORE_METRICS` | 5 項核心指標 | 非頻譜的關鍵特徵 |
| `PCA n_components` | `0.90` | 保留變異比例 |
| `IsolationForest contamination` | `0.05` | 預期異常比例 |
| 樣本數門檻 | `300` | 群組樣本不足則不建模 |

---

## 🇺🇸 English

### Overview

An **unsupervised anomaly detection pipeline** for acoustic measurements collected from industrial production lines. The pipeline:

1. Converts raw CSV / Excel measurement files to **Parquet** to speed up downstream I/O.
2. Builds models **per (`Acoustic_Family`, `Line_Name`, `Device_ID`) group** so different machine types don't contaminate each other's baselines.
3. Reduces dimensionality on 14 one-third-octave bands (500 Hz – 10 kHz) using **RobustScaler + PCA** (90 % variance retained).
4. Concatenates the reduced spectral features with core metrics (`dB(A)`, `index1`, `index2`, `index3`, `RPM`).
5. Flags outliers with **Isolation Forest** (`contamination=0.05`) and emits an `Anomaly_Score` for ranking.

### Data Requirements

Each input file (CSV or XLSX) must contain at least:

| Category | Columns |
| --- | --- |
| Identifiers | `Model_Name`, `Line_Name`, `Device_ID` |
| Core metrics | `dB(A)`, `index1`, `index2`, `index3`, `RPM` |
| Spectrum | `500.0`, `630.0`, `800.0`, `1000.0`, `1250.0`, `1600.0`, `2000.0`, `2500.0`, `3150.0`, `4000.0`, `5000.0`, `6300.0`, `8000.0`, `10000.0` |

> Groups with fewer than **300** samples are skipped to avoid unstable models on small populations.

### Installation

```bash
git clone https://github.com/Seilin876/acoustic-anomaly-detection.git
cd acoustic-anomaly-detection
pip install -r requirements.txt
```

### Usage

1. Drop your `.csv` / `.xlsx` files into the project root (or edit `SOURCE_DIR`).
2. **Configure `model_groups.txt`** — list which `Model_Name` values to analyze and how to group them:

   ```
   # Models in the same group are treated as one acoustic family
   # Different groups are modelled separately; unlisted models are skipped
   Family_FFB = FFB0412UHN-CH68, FFB0412UHNFFE, FFB0412UHNFSW B
   Family_TAA = TAA0412CD-AF83 A, TAA0412DDX01VXW F
   ```

3. Run:

   ```bash
   python anomaly_detection.py
   ```

4. Outputs:
   - `./output_parquet/`: Parquet cache of the raw input files.
   - `Anomaly_Detection_Results.csv`: rows flagged as anomalies (`Anomaly_Flag == -1`), enriched with `Anomaly_Score`, `PCA_Components`, and `Acoustic_Family` (the group name you set in the config).

### Configuration

All knobs live at the top of `anomaly_detection.py`:

| Parameter | Default | Description |
| --- | --- | --- |
| `SOURCE_DIR` | `./` | Folder containing the raw files |
| `OUTPUT_DIR` | `./output_parquet` | Where Parquet caches are written |
| `RESULTS_CSV` | `Anomaly_Detection_Results.csv` | Output file name |
| `MODEL_GROUPS_FILE` | `model_groups.txt` | Path to the model-groups config file |
| `FREQ_BANDS` | 14 one-third-octave bands | Spectrum column names |
| `CORE_METRICS` | 5 core metrics | Non-spectral features |
| `PCA n_components` | `0.90` | Variance ratio to retain |
| `IsolationForest contamination` | `0.05` | Expected anomaly fraction |
| Min samples per group | `300` | Skip groups smaller than this |

---

## Pipeline at a glance

```
raw .csv / .xlsx                  model_groups.txt
       │                                 │
       ▼                                 ▼
 convert_to_parquet()  ──►  filter & map Model_Name → Acoustic_Family
       │                                 │
       └──────────────►◄─────────────────┘
                       │
                       ▼
       groupby(Acoustic_Family, Line_Name, Device_ID)
                       │
       ├── FREQ_BANDS  ──► RobustScaler ──► PCA(0.90)  ┐
       │                                                ├── concat ──► IsolationForest
       └── CORE_METRICS ──► RobustScaler ──────────────┘            │
                                                                     ▼
                                                       Anomaly_Detection_Results.csv
```

## Requirements

- Python 3.9+
- `pandas`, `scikit-learn`, `pyarrow`, `openpyxl`

See `requirements.txt`.

## License

[MIT](./LICENSE)
