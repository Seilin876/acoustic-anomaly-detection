import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# sklearn PCA 在內部 SVD/matmul 路徑會偶發 RuntimeWarning（已知的 numpy FP 旗標互動，
# 結果仍正確）。輸入已用 nan_to_num 防禦過 inf/NaN，這裡僅抑制噪音。
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.decomposition\._base")

# --- 參數設定 (Configuration) ---
SOURCE_DIR = "./"  # 替換為您的 Excel 資料夾路徑
OUTPUT_DIR = "./output_parquet"
RESULTS_CSV = "Anomaly_Detection_Results.csv"
MODEL_GROUPS_FILE = "model_groups.txt"  # 機種分組設定檔

FREQ_BANDS = ['500.0', '630.0', '800.0', '1000.0', '1250.0', '1600.0',
              '2000.0', '2500.0', '3150.0', '4000.0', '5000.0', '6300.0',
              '8000.0', '10000.0']
CORE_METRICS = ['dB(A)', 'index1', 'index2', 'index3', 'RPM']


def load_model_groups(config_path):
    """讀取 'GroupName = Model1, Model2, ...' 格式的設定檔，回傳 {model: group} 字典。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"設定檔 '{config_path}' 不存在。請建立此檔並用 "
            f"'GroupName = Model1, Model2' 格式列出要分析的機種群組。"
        )

    mapping = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                print(f"  Warning: 第 {lineno} 行無 '=' 符號，已略過: {line!r}")
                continue
            group, models = line.split('=', 1)
            group = group.strip()
            models = [m.strip() for m in models.split(',') if m.strip()]
            if not group or not models:
                print(f"  Warning: 第 {lineno} 行群組名或機種列表為空，已略過")
                continue
            for m in models:
                if m in mapping and mapping[m] != group:
                    print(f"  Warning: 機種 '{m}' 同時出現在 '{mapping[m]}' 與 '{group}'，"
                          f"以 '{group}' 為準")
                mapping[m] = group

    if not mapping:
        raise ValueError(
            f"設定檔 '{config_path}' 沒有定義任何群組。"
            f"請至少加入一行 'GroupName = Model1, Model2'。"
        )

    n_groups = len(set(mapping.values()))
    print(f"Loaded {len(mapping)} models across {n_groups} group(s) from {config_path}:")
    for group_name in sorted(set(mapping.values())):
        members = [m for m, g in mapping.items() if g == group_name]
        print(f"  - {group_name}: {', '.join(members)}")
    return mapping

def convert_to_parquet(source_dir, output_dir):
    """將 Excel 或 CSV 轉換為 Parquet 以提升後續 I/O 效能"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支援讀取您提供的 csv 或 xlsx
    files = glob.glob(os.path.join(source_dir, "*.csv")) + glob.glob(os.path.join(source_dir, "*.xlsx"))

    parquet_paths = []
    for file in files:
        base_name = os.path.basename(file).split('.')[0] + '.parquet'
        out_path = os.path.join(output_dir, base_name)
        parquet_paths.append(out_path)

        if os.path.exists(out_path):
            continue

        print(f"Converting: {file} -> Parquet")
        if file.endswith('.csv'):
            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        else:
            df = pd.read_excel(file, sheet_name=0)

        # 1. 強制欄位名稱為字串
        df.columns = df.columns.astype(str)

        # 2. 強制解決 PyArrow Schema 型別衝突 (Type Coercion)
        # 產線資料常有髒數據，將所有物件型態強制轉為字串，滿足 Parquet 儲存規範
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        df.to_parquet(out_path, engine='pyarrow', compression='snappy')

    return parquet_paths

def run_anomaly_pipeline(parquet_files, model_to_group):
    """執行 PCA + Isolation Forest 管線"""
    all_results = []

    for file in parquet_files:
        print(f"Processing {file}...")
        df = pd.read_parquet(file)

        # 依設定檔將 Model_Name 映射到 Acoustic_Family (群組名)
        df['Acoustic_Family'] = df['Model_Name'].astype(str).map(model_to_group)
        skipped = df[df['Acoustic_Family'].isna()]
        if len(skipped):
            unmapped = sorted(skipped['Model_Name'].astype(str).unique())
            print(f"  Skipped {len(skipped)} rows from {len(unmapped)} unconfigured "
                  f"model(s): {unmapped}")
        df = df.dropna(subset=['Acoustic_Family'])
        if df.empty:
            print(f"  No rows match the configured model groups in {file}; skipping.")
            continue

        # 分層處理
        grouped = df.groupby(['Acoustic_Family', 'Line_Name', 'Device_ID'])

        for (family, line, device), group in grouped:
            if len(group) < 300: # 樣本數不足 300 拒絕建模
                continue

            # 1. 頻譜特徵 PCA
            X_freq = group[FREQ_BANDS].apply(pd.to_numeric, errors='coerce').fillna(0)
            scaler_freq = RobustScaler()
            X_freq_scaled = scaler_freq.fit_transform(X_freq)
            # 群組內常數欄位會讓 RobustScaler 產生 inf (IQR=0)，需替換為 0 避免 PCA 數值爆掉
            X_freq_scaled = np.nan_to_num(X_freq_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            pca = PCA(n_components=0.90, random_state=42) # 保留 90% 變異
            X_pca = pca.fit_transform(X_freq_scaled)
            df_pca = pd.DataFrame(X_pca, index=group.index)

            # 2. 核心特徵標準化
            X_core = group[CORE_METRICS].apply(pd.to_numeric, errors='coerce').fillna(0)
            scaler_core = RobustScaler()
            X_core_scaled = scaler_core.fit_transform(X_core)
            X_core_scaled = np.nan_to_num(X_core_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            df_core_scaled = pd.DataFrame(X_core_scaled, index=group.index)

            # 3. 特徵融合與模型訓練
            X_final = pd.concat([df_core_scaled, df_pca], axis=1)
            X_final.columns = X_final.columns.astype(str)

            clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            group_copy = group.copy()
            group_copy['Anomaly_Flag'] = clf.fit_predict(X_final) # -1 為異常，1 為正常
            group_copy['Anomaly_Score'] = clf.decision_function(X_final)
            group_copy['PCA_Components'] = X_pca.shape[1]

            # 僅保留判定為異常的資料匯出
            anomalies = group_copy[group_copy['Anomaly_Flag'] == -1]
            all_results.append(anomalies)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
        print(f"Done. Detected {len(final_df)} anomalies. Results saved to {RESULTS_CSV}")
    else:
        print("No models met the sample size threshold or no anomalies detected.")

if __name__ == "__main__":
    model_to_group = load_model_groups(MODEL_GROUPS_FILE)
    parquet_files = convert_to_parquet(SOURCE_DIR, OUTPUT_DIR)
    run_anomaly_pipeline(parquet_files, model_to_group)
