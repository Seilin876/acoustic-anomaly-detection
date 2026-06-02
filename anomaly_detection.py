import os
import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# --- 參數設定 (Configuration) ---
SOURCE_DIR = "./"  # 替換為您的 Excel 資料夾路徑
OUTPUT_DIR = "./output_parquet"
RESULTS_CSV = "Anomaly_Detection_Results.csv"

FREQ_BANDS = ['500.0', '630.0', '800.0', '1000.0', '1250.0', '1600.0',
              '2000.0', '2500.0', '3150.0', '4000.0', '5000.0', '6300.0',
              '8000.0', '10000.0']
CORE_METRICS = ['dB(A)', 'index1', 'index2', 'index3', 'RPM']

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

        df.columns = df.columns.astype(str)
        df.to_parquet(out_path, engine='pyarrow', compression='snappy')

    return parquet_paths

def run_anomaly_pipeline(parquet_files):
    """執行 PCA + Isolation Forest 管線"""
    all_results = []

    for file in parquet_files:
        print(f"Processing {file}...")
        df = pd.read_parquet(file)

        # 建立聲學家族映射 (此處簡化，您可依需求載入 JSON)
        df['Acoustic_Family'] = df['Model_Name'].astype(str)

        # 分層處理
        grouped = df.groupby(['Acoustic_Family', 'Line_Name', 'Device_ID'])

        for (family, line, device), group in grouped:
            if len(group) < 300: # 樣本數不足 300 拒絕建模
                continue

            # 1. 頻譜特徵 PCA
            X_freq = group[FREQ_BANDS].apply(pd.to_numeric, errors='coerce').fillna(0)
            scaler_freq = RobustScaler()
            X_freq_scaled = scaler_freq.fit_transform(X_freq)

            pca = PCA(n_components=0.90, random_state=42) # 保留 90% 變異
            X_pca = pca.fit_transform(X_freq_scaled)
            df_pca = pd.DataFrame(X_pca, index=group.index)

            # 2. 核心特徵標準化
            X_core = group[CORE_METRICS].apply(pd.to_numeric, errors='coerce').fillna(0)
            scaler_core = RobustScaler()
            X_core_scaled = scaler_core.fit_transform(X_core)
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
    parquet_files = convert_to_parquet(SOURCE_DIR, OUTPUT_DIR)
    run_anomaly_pipeline(parquet_files)
