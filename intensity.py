import os
import re
import csv
import numpy as np
"""
提取PGA并计算烈度 / Extract PGA and calculate intensity
"""
# ==== 烈度计算公式 / Intensity calculation formula ====
def compute_intensity_components(pga, pgv):
    """
    根据PGA和PGV计算烈度分量 / Calculate intensity components from PGA and PGV
    
    参数 / Parameters:
        pga: 峰值地面加速度 (cm/s²) / Peak ground acceleration (cm/s²)
        pgv: 峰值地面速度 (cm/s) / Peak ground velocity (cm/s)
    
    返回 / Returns:
        I_A: 基于加速度的烈度 / Acceleration-based intensity
        I_V: 基于速度的烈度 / Velocity-based intensity
        I: 综合烈度 / Combined intensity
    """
    if pga <= 0 or pgv <= 0:
        return None, None, None
    I_A = 3.17 * np.log10(pga / 100) + 6.59  # 转为 g / Convert to g
    I_V = 3.00 * np.log10(pgv / 100) + 9.77
    if I_A >= 6.0 and I_V >= 6.0:
        I = round((I_A + I_V) / 2, 1)
    else:
        I = round(min(I_A, I_V), 1)
    return round(I_A, 2), round(I_V, 2), I

# 主目录路径 / Main directory paths
main_folders = {
    # r"G:\exsim\dianxibei\1955M7.5\112",
# "G:\exsim\dianxibei\1955M7.5\211",
r"D:\PYTHON\exsim_manual_717\1981（已bx）"
    # r"G:\exsim\sample\dizhenyanjiu\seg_130bar\62",
# r"G:\exsim\sample\dizhenyanjiu\seg_130bar\36"
    # r"G:\exsim\sample\dizhenyanjiu\seg\62_re"

    # "G:\exsim\sample\dizhenyanjiu\seg\68"
    # r"F:\all\code\2月26\68"
    # r"G:\exsim\sample\dizhenyanjiu\rup_cen\up"
}

# 正则表达式 / Regular expressions
acc_pattern = re.compile(r"maximum absolute acceleration:\s+([\d.]+)")
acc_site_pattern = re.compile(r"_acc_s(\d{3})")
psa_site_pattern = re.compile(r"Site\s+#\s+(\d+)")
coord_pattern = re.compile(r"Site coordinates:\s+([\d.E+-]+)\s+([\d.E+-]+)")
psa_data_line_pattern = re.compile(
    r"^\s*([\d\.\-E+]+)\s+([\d\.\-E+]+)\s+([\d\.\-E+]+)?\s+([\d\.\-E+]+)?\s+([\d\.\-E+]+)?"
)

# 遍历目录 / Traverse directories
for main_folder in main_folders:
    results = []

    for root, dirs, files in os.walk(main_folder):
        acc_data = {}
        pgv_data = {}

        # 提取 PGA / Extract PGA
        acc_files = [f for f in files if "_acc_s" in f and f.endswith(".out")]
        for acc_file in acc_files:
            path = os.path.join(root, acc_file)
            site_match = acc_site_pattern.search(acc_file)
            if not site_match:
                continue
            site_id = site_match.group(1)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                acc_match = acc_pattern.search(content)
                if acc_match:
                    try:
                        pga = float(acc_match.group(1))
                        acc_data[site_id] = pga
                    except ValueError:
                        continue

        # 提取 PGV（T=1.0s 的 PSV × 2π） + 经纬度 / Extract PGV (PSV at T=1.0s × 2π) + latitude/longitude
        psa_files = [f for f in files if "_psa_fs_s" in f and f.endswith(".out")]
        for psa_file in psa_files:
            path = os.path.join(root, psa_file)
            site_id = None
            lat = None
            lon = None
            psv_1s = None  # ← 重置位置正确 / Reset position correct

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if site_id is None:
                        match = psa_site_pattern.search(line)
                        if match:
                            site_id = match.group(1).zfill(3)

                    if lat is None and "Site coordinates" in line:
                        coord_match = coord_pattern.search(line)
                        if coord_match:
                            lat = float(coord_match.group(1))
                            lon = float(coord_match.group(2))

                    data_match = psa_data_line_pattern.match(line)
                    if data_match:
                        try:
                            period = float(data_match.group(2))
                            psv_val = data_match.group(5)
                            if psv_val and abs(period - 1.0) < 0.01:
                                psv_1s = float(psv_val)
                        except ValueError:
                            continue

            if site_id and lat is not None and lon is not None and psv_1s is not None:
                pgv = 2 * np.pi * psv_1s  # 转为 PGV / Convert to PGV
                pgv_data[site_id] = (lat, lon, pgv)

        # 合并数据 / Merge data
        for site_id in sorted(pgv_data.keys()):
            lat, lon, pgv = pgv_data[site_id]
            pga = acc_data.get(site_id, None)
            if pga is not None:
                I_A, I_V, I = compute_intensity_components(pga, pgv)
                results.append([
                    site_id, lat, lon,
                    f"{pga:.2f}", f"{pgv:.2f}",
                    f"{I_A:.2f}", f"{I_V:.2f}", f"{I:.1f}",
                    os.path.basename(root)
                ])

    # 输出结果 / Output results
    output_file = os.path.join(main_folder, "test627.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not results:
        print(f"❗{main_folder} 未提取到任何数据，请检查格式 / No data extracted, please check format")
    else:
        with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Site_ID", "Latitude", "Longitude",
                "PGA (cm/s^2)", "PGV (cm/s)",
                "I_A", "I_V", "Intensity", "Subfolder"
            ])
            writer.writerows(results)
        print(f"✅ {main_folder} 提取完成，保存为：{output_file} / Extraction completed, saved as: {output_file}")
