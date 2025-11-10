# Earthquake Simulation Analysis Toolset

## Introduction

This repository hosts a collection of utilities for simulating earthquakes and analysing intensity distributions:

- Generate EXSIM parameter files and manage station coordinates.
- Build simplified asperity (slip) models.
- Extract PGA/PGV values from EXSIM outputs and convert them to intensity.
- Create comprehensive intensity visualisations combining historical data, simulated contours, cities, and faults.
- Automate EXSIM batch execution.

## Files

- **1. `main.py` – EXSIM parameter generator**
  - Load `.params` templates, edit selected fields, and create station grids within a latitude/longitude window.
  - Split stations into groups for parallel EXSIM runs and produce per-group parameter/support files.
  - Usage:
    ```bash
    python main.py
    ```

- **2. `generate_asperity_simple.py` – asperity model builder**
  - Construct simplified asperity distributions (size = 1/3 fault, depth 5–15 km).
  - Supports positioning asperities (left/centre/right), enforces seismic moment balance, and exports slip weights plus plots.
  - Usage:
    ```bash
    python generate_asperity_simple.py
    ```

- **3. `intensity.py` – PGA/PGV extraction and intensity calculation**
  - Parse EXSIM acceleration (`*_acc_s*.out`) and PSA (`*_psa_fs_s*.out`) files to obtain PGA/PGV.
  - Compute intensity values (default formula: `IA = 3.17 log10(PGA/100) + 6.59`, `IV = 3.00 log10(PGV/100) + 9.77`) and export CSV summaries.
  - Configure `main_folders` before running:
    ```bash
    python intensity.py
    ```

- **4. `earthquake_analysis_combined.py` – comprehensive intensity plotting tool**
  - Combine historical isoseismals, simulated intensity CSVs, city points, and fault traces into a single figure.
  - GUI workflow: select historical Excel, city Excel, fault Excel, then one/multiple simulation CSVs, set output title/file name, choose legend position, optionally limit bounding box, and generate the plot.

- **4.1 `amplify_pga_pgv.py` – site amplification & interpolated intensity**
  - Input data:
    - Site catalogue (`NEHRP/Quaternary_YGeology_NEHRP.shp`) providing NEHRP site classes or VS30 (default to class B if missing).
    - Eight ground-motion CSV files in `PGAPGV/` containing latitude, longitude, PGA (Gal) and PGV (cm/s).
  - Processing steps:
    - Discover supported site & motion files automatically.
    - Map each station to the nearest site using a KDTree (initial radius 2 km; auto-expand to 5 km if >50% fallback).
    - Apply Chinese mainland amplification factors (B:1.0/1.0, C:1.2/1.2, D:1.8/1.5, E:2.0/1.7).
    - Convert to m/s² and m/s before computing intensity:
      ```
      IPGA = 3.17 × log10(PGA_amp / 100) + 6.59
      IPGV = 3.17 × log10(PGV_amp / 100) + 9.77
      I = IPGV (if both ≥ 6); otherwise I = 0.5 × (IPGA + IPGV); capped at 11.
      ```
    - Interpolate the amplified intensity field on a regular grid (`scipy.interpolate.griddata`), clip outside the convex hull, and draw 0.5-level contours.
  - Usage:
    ```bash
    py amplify_pga_pgv.py --site_dir NEHRP --gm_dir PGAPGV --out_dir out
    ```
  - Outputs (per event and combined):
    - `out/<event>_amplified.csv` – original/amp PGA/PGV, Fa/Fv, IPGA, IPGV, final intensity, match distance.
    - `out/<event>_intensity.png` – interpolated intensity contour map.
    - `out/all_events_amplified.csv` – concatenated results.
    - `out/summary.txt` – event-wise statistics (sample counts, means, maxima, intensity percentiles).

- **5. `autobx.bat` – EXSIM batch runner**
  - Recursively traverse subdirectories, launch `exsim_dmb.exe` where appropriate, and open each run in a dedicated console window.

## Dependencies

- `tkinter`, `numpy`, `matplotlib`, `pandas`, `scipy`, `openpyxl`
- Install:
  ```bash
  pip install numpy matplotlib pandas scipy openpyxl
  ```

## Typical workflow

1. Generate asperity model  
   `python generate_asperity_simple.py`
2. Create EXSIM parameter files  
   `python main.py`
3. Batch-run EXSIM  
   `autobx.bat`
4. Extract PGA/PGV and compute intensity  
   `python intensity.py`
5. Produce integrated visuals  
   `python earthquake_analysis_combined.py`

## Notes

- Update hard-coded paths as needed (`main.py`, `intensity.py`, `earthquake_analysis_combined.py`).
- Ensure input files match expected schemas (column names, data types, `.xlsx` for Excel).
- Validate latitude/longitude ranges and general data integrity.
- Large station sets can be slow—use grouping or parallel execution to improve throughput.

## Contact

- Author: Chaohui Feng  
- Email: fengchaohui23@mails.ucas.ac.cn  
- Date: 2025-10-01

## License

Academic research use only.

---

# 地震模拟分析工具集（中文版）

## 简介

本仓库提供一组用于地震模拟与烈度分析的脚本，包括：

- 批量生成 EXSIM 参数文件与站点坐标。
- 构建简化的凹凸体滑移模型。
- 从 EXSIM 输出中提取 PGA/PGV 并计算烈度。
- 联合历史等震线、模拟结果、城市、断层制作综合烈度图。
- 自动化批量执行 EXSIM。

## 主要文件

- **1. `main.py` – EXSIM 多方案参数生成器**
  - 加载 `.params` 模板、修改选定字段、并在给定经纬度范围内生成站点。
  - 支持将站点划分为多组，分别生成参数/支撑文件以便并行运行。

- **2. `generate_asperity_simple.py` – 凹凸体模型生成**
  - 构建尺寸为断层 1/3（深度 5–15 km）的简化凹凸体，支持左/中/右位置。
  - 保证总地震矩守恒，输出滑移权重及可视化图像。

- **3. `intensity.py` – PGA/PGV 提取与烈度计算**
  - 解析 `*_acc_s*.out`、`*_psa_fs_s*.out` 文件，提取 PGA/PGV。
  - 默认公式：`IA = 3.17 log10(PGA/100) + 6.59`, `IV = 3.00 log10(PGV/100) + 9.77`。
  - 运行前请修改 `main_folders`。

- **4. `earthquake_analysis_combined.py` – 地震烈度综合分析工具**
  - 读取历史等震线、城市、断层 Excel 以及模拟 CSV。
  - 图形界面按步骤选择文件、设置标题/输出文件名、可选裁剪范围，生成综合分析图。

- **4.1 `amplify_pga_pgv.py` – 场地放大与烈度插值**
  - 数据源：
    - `NEHRP/Quaternary_YGeology_NEHRP.shp` 提供 NEHRP 场地类别或 VS30；缺失时默认 B 类。
    - `PGAPGV/*.csv` 包含 8 个事件的站点 PGA（Gal）和 PGV（cm/s）。
  - 核心流程：
    - 自动发现场地与地震动文件，构建 KDTree 做最近邻匹配（默认 2 km，退化比例 >50% 时自动改 5 km）。
    - 使用中国大陆典型放大系数：B 1.0/1.0、C 1.2/1.2、D 1.8/1.5、E 2.0/1.7。
    - 将放大后的 PGA、PGV 换算为 SI 单位，按公式计算烈度：
      ```
      IPGA = 3.17 × log10(PGA_amp / 100) + 6.59
      IPGV = 3.17 × log10(PGV_amp / 100) + 9.77
      I = IPGV (IPGA ≥ 6 且 IPGV ≥ 6，否则 I = 0.5 × (IPGA + IPGV))
      ```
    - 利用 `scipy.interpolate.griddata` 在凸包内插值，绘制 0.5 度间隔的烈度等值线。
  - 使用方法：
    ```bash
    py amplify_pga_pgv.py --site_dir NEHRP --gm_dir PGAPGV --out_dir out
    ```
  - 输出文件：
    - `out/<event>_amplified.csv` – 原始/放大 PGA/PGV、Fa/Fv、IPGA、IPGV、烈度、匹配距离。
    - `out/<event>_intensity.png` – 放大烈度等值线图。
    - `out/all_events_amplified.csv` – 所有事件汇总。
    - `out/summary.txt` – 样本数、放大前后均值/极值、烈度分位数等统计。

- **5. `autobx.bat` – EXSIM 批量执行脚本**
  - 递归遍历目录，找到含 EXSIM 模型的最末级文件夹并运行 `exsim_dmb.exe`，每个实例独立窗口执行。

## 依赖

- 所需库：`tkinter`、`numpy`、`matplotlib`、`pandas`、`scipy`、`openpyxl`
- 安装命令：
  ```bash
  pip install numpy matplotlib pandas scipy openpyxl
  ```

## 典型流程

1. 生成凹凸体模型 – `python generate_asperity_simple.py`
2. 生成 EXSIM 参数 – `python main.py`
3. 批量运行 EXSIM – `autobx.bat`
4. 提取 PGA/PGV 并计算烈度 – `python intensity.py`
5. 绘制综合烈度图 – `python earthquake_analysis_combined.py`

## 注意事项

- 请根据实际环境修改脚本中的硬编码路径。
- 确认输入文件格式正确，Excel 建议使用 `.xlsx`。
- 检查经纬度等基础数据是否合理。
- 站点数量较多时，计算时间较长，可通过分组/并行方式提高效率。

## 联系方式

- 作者：奉超晖  
- 邮箱：fengchaohui23@mails.ucas.ac.cn  
- 日期：2025-10-01

## 许可证

本项目仅供学术研究使用。