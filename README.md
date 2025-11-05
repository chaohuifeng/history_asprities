# 地震模拟分析工具集 / Earthquake Simulation Analysis Toolset

## 简介 / Introduction

本项目包含一套用于地震模拟和烈度分析的工具集，主要用于：
This project contains a toolset for earthquake simulation and intensity analysis, mainly used for:

- EXSIM参数文件批量生成和站点坐标管理
- EXSIM parameter file batch generation and site coordinate management
- 地震凹凸体模型生成
- Earthquake asperity model generation
- 从EXSIM输出文件中提取PGA/PGV并计算烈度
- Extract PGA/PGV from EXSIM output files and calculate intensity
- 地震烈度综合分析可视化
- Comprehensive earthquake intensity analysis visualization
- EXSIM批量执行脚本
- EXSIM batch execution script

## 文件说明 / File Description

### 1. main.py
**功能 / Function:** EXSIM参数生成器 / EXSIM Parameter Generator

**用途 / Usage:**
- 加载EXSIM参数文件（.params格式）
- Load EXSIM parameter files (.params format)
- 修改指定行的参数值
- Modify parameter values in specified lines
- 生成指定经纬度范围内的站点坐标
- Generate site coordinates within specified latitude/longitude range
- 将站点分组为多个数组，支持并行处理
- Group sites into multiple arrays for parallel processing
- 为每个数组生成独立的参数文件和必要的支持文件
- Generate independent parameter files and necessary support files for each array

**使用方法 / How to Use:**
```bash
python main.py
```

1. 点击"浏览"按钮选择参数文件
2. Click "Browse" button to select parameter file
3. 修改界面中显示的基础参数（如需要）
4. Modify basic parameters shown in interface (if needed)
5. 设置站点经纬度范围和步长
6. Set site latitude/longitude range and step size
7. 设置分组数量（用于并行处理）
8. Set number of groups (for parallel processing)
9. 输入输出路径
10. Enter output path
11. 点击"保存并生成"按钮
12. Click "Save and Generate" button

**输出 / Output:**
- 在输出路径下创建多个子文件夹（Array_1, Array_2, ...）
- Create multiple subfolders under output path (Array_1, Array_2, ...)
- 每个子文件夹包含修改后的参数文件和必要的支持文件
- Each subfolder contains modified parameter file and necessary support files

---

### 2. generate_asperity_simple.py
**功能 / Function:** 地震凹凸体模型生成器（简化版）/ Earthquake Asperity Model Generator (Simplified Version)

**用途 / Usage:**
- 生成指定大小的地震凹凸体模型
- Generate earthquake asperity models of specified size
- 凹凸体大小为断层的1/3，深度范围5-15km
- Asperity size is 1/3 of fault, depth range 5-15km
- 凹凸体滑移量占比0.5，背景滑移量占比0.25
- Asperity slip ratio 0.5, background slip ratio 0.25
- 支持凹凸体位置控制（左、中、右）
- Support asperity position control (left, center, right)
- 保持能量平衡，确保总地震矩正确
- Maintain energy balance, ensure correct total seismic moment
- 输出滑移分布文件和可视化图片
- Output slip distribution files and visualization plots

**使用方法 / How to Use:**
```bash
python generate_asperity_simple.py
```

**参数设置 / Parameter Settings:**
在文件顶部的参数设置区域修改以下参数：
Modify the following parameters in the parameter setting area at the top of the file:

```python
Mw = 7.0          # 矩震级 / Moment magnitude
L = 64.0          # 断层长度 (km) / Fault length (km)
W = 20.0           # 断层宽度 (km) / Fault width (km)
dx = 2.0           # 子断层长度 (km) / Subfault length (km)
dz = 2.0           # 子断层宽度 (km) / Subfault width (km)
positions = ['left', 'center', 'right']  # 凹凸体位置 / Asperity positions
generate_plots = True  # 是否生成可视化图片 / Whether to generate plots
save_slip_files = True  # 是否保存滑移权重文件 / Whether to save slip weight files
```

**输出 / Output:**
- `Mw{Mw}_L{L}km_W{W}km_{position}_slip_weights.txt` - 滑移权重文件
- `Mw{Mw}_L{L}km_W{W}km_{position}_slip_weights.txt` - Slip weight file
- `Mw{Mw}_L{L}km_W{W}km_{position}_plots.png` - 可视化图片
- `Mw{Mw}_L{L}km_W{W}km_{position}_plots.png` - Visualization plot

---

### 3. intensity.py
**功能 / Function:** PGA/PGV提取和烈度计算 / PGA/PGV Extraction and Intensity Calculation

**用途 / Usage:**
- 从EXSIM输出文件中提取峰值地面加速度（PGA）
- Extract Peak Ground Acceleration (PGA) from EXSIM output files
- 从PSA文件中提取峰值地面速度（PGV）
- Extract Peak Ground Velocity (PGV) from PSA files
- 根据PGA和PGV计算地震烈度
- Calculate earthquake intensity from PGA and PGV
- 输出包含站点ID、经纬度、PGA、PGV和烈度的CSV文件
- Output CSV file containing site ID, latitude/longitude, PGA, PGV, and intensity

**使用方法 / How to Use:**
1. 修改文件中的 `main_folders` 变量，设置要处理的目录路径
2. Modify the `main_folders` variable in the file to set directory paths to process
3. 运行脚本：
4. Run the script:

```bash
python intensity.py
```

**输入文件格式要求 / Input File Format Requirements:**
- 加速度文件：`*_acc_s*.out` - 包含 "maximum absolute acceleration:" 行
- Acceleration files: `*_acc_s*.out` - Contains "maximum absolute acceleration:" line
- PSA文件：`*_psa_fs_s*.out` - 包含站点坐标和周期为1.0s的PSV值
- PSA files: `*_psa_fs_s*.out` - Contains site coordinates and PSV value at period 1.0s

**输出 / Output:**
- `test627.csv` - 包含所有提取的数据和计算的烈度值
- `test627.csv` - Contains all extracted data and calculated intensity values

**烈度计算公式 / Intensity Calculation Formula:**
- I_A = 3.17 × log₁₀(PGA/100) + 6.59
- I_V = 3.00 × log₁₀(PGV/100) + 9.77
- I = (I_A + I_V)/2 (当 I_A ≥ 6.0 且 I_V ≥ 6.0)
- I = min(I_A, I_V) (其他情况)

---

### 4. earthquake_analysis_combined.py
**功能 / Function:** 地震烈度综合分析工具 / Comprehensive Earthquake Intensity Analysis Tool

**用途 / Usage:**
- 读取历史等震线数据并绘制
- Read and plot historical isoseismal data
- 读取城市点数据并标注
- Read and annotate city point data
- 读取断层数据并绘制断层线
- Read and plot fault data
- 读取模拟CSV文件并绘制模拟烈度等值线
- Read simulation CSV files and plot simulated intensity contours
- 生成综合分析图，包含历史烈度、模拟烈度、城市点和断层信息
- Generate comprehensive analysis plot including historical intensity, simulated intensity, city points, and fault information

**使用方法 / How to Use:**
```bash
python earthquake_analysis_combined.py
```

**操作步骤 / Steps:**
1. 第一步：选择历史等震线数据Excel文件（需包含列：intensity, Longtitude, Latitude）
2. Step 1: Select historical isoseismal data Excel file (must contain columns: intensity, Longtitude, Latitude)
3. 选择城市点数据Excel文件（需包含列：city_name, Longtitude, Latitude）
4. Select city point data Excel file (must contain columns: city_name, Longtitude, Latitude)
5. 选择断层数据Excel文件（需包含列：Longtitude, Latitude）
6. Select fault data Excel file (must contain columns: Longtitude, Latitude)
7. 第二步：选择一个或多个模拟CSV文件（需包含列：Longitude, Latitude, Intensity）
8. Step 2: Select one or more simulation CSV files (must contain columns: Longitude, Latitude, Intensity)
9. 输入图片标题和输出文件名
10. Enter plot title and output filename
11. 选择图例位置（左下角或右上角）
12. Select legend position (lower left or upper right)
13. （可选）启用范围裁剪，设置经纬度范围
14. (Optional) Enable range clipping, set latitude/longitude range
15. 点击"生成综合分析图"按钮
16. Click "Generate Comprehensive Analysis Plot" button

**输出 / Output:**
- 图片保存在：`D:\PYTHON\exsim_manual_717\生成图像\{输出文件名}.png`
- Image saved to: `D:\PYTHON\exsim_manual_717\生成图像\{output_filename}.png`
- 高分辨率PNG格式（300 DPI）
- High resolution PNG format (300 DPI)

---

### 5. autobx.bat
**功能 / Function:** EXSIM批量执行脚本 / EXSIM Batch Execution Script

**用途 / Usage:**
- 递归遍历指定目录下的所有子目录
- Recursively traverse all subdirectories under specified directory
- 在每个没有更深子目录的文件夹中运行exsim_dmb.exe
- Run exsim_dmb.exe in each folder without deeper subdirectories
- 自动在新窗口中启动每个EXSIM实例
- Automatically start each EXSIM instance in a new window

**使用方法 / How to Use:**
1. 将脚本放在包含EXSIM运行目录的根目录下
2. Place the script in the root directory containing EXSIM run directories
3. 双击运行脚本，或在命令行中执行：
4. Double-click to run the script, or execute in command line:

```bash
autobx.bat
```

**注意事项 / Notes:**
- 确保每个子目录中都包含 `exsim_dmb.exe` 和 `exsim_dmb.params` 文件
- Ensure each subdirectory contains `exsim_dmb.exe` and `exsim_dmb.params` files
- 脚本会在新窗口中启动每个EXSIM实例，可以并行运行
- Script will start each EXSIM instance in a new window, allowing parallel execution

---

## 依赖库 / Dependencies

### Python库 / Python Libraries
- `tkinter` - GUI界面（通常随Python安装）
- `tkinter` - GUI interface (usually installed with Python)
- `numpy` - 数值计算
- `numpy` - Numerical computing
- `matplotlib` - 绘图
- `matplotlib` - Plotting
- `pandas` - 数据处理
- `pandas` - Data processing
- `scipy` - 科学计算
- `scipy` - Scientific computing
- `openpyxl` - Excel文件读取（用于pandas）
- `openpyxl` - Excel file reading (for pandas)

### 安装依赖 / Install Dependencies
```bash
pip install numpy matplotlib pandas scipy openpyxl
```

---

## 工作流程 / Workflow

### 典型工作流程 / Typical Workflow

1. **生成凹凸体模型** / **Generate Asperity Model**
   ```bash
   python generate_asperity_simple.py
   ```
   - 生成滑移权重文件（slip_weights.txt）
   - Generate slip weight files (slip_weights.txt)

2. **生成EXSIM参数文件** / **Generate EXSIM Parameter Files**
   ```bash
   python main.py
   ```
   - 为多个站点组生成参数文件
   - Generate parameter files for multiple site groups

3. **批量运行EXSIM** / **Batch Run EXSIM**
   ```bash
   autobx.bat
   ```
   - 并行运行所有EXSIM实例
   - Run all EXSIM instances in parallel

4. **提取烈度数据** / **Extract Intensity Data**
   ```bash
   python intensity.py
   ```
   - 从EXSIM输出文件中提取PGA/PGV并计算烈度
   - Extract PGA/PGV from EXSIM output files and calculate intensity

5. **生成综合分析图** / **Generate Comprehensive Analysis Plot**
   ```bash
   python earthquake_analysis_combined.py
   ```
   - 可视化历史烈度、模拟烈度和断层信息
   - Visualize historical intensity, simulated intensity, and fault information

---

## 注意事项 / Notes

1. **文件路径** / **File Paths**
   - 某些脚本中包含硬编码的路径，使用前请根据实际情况修改
   - Some scripts contain hardcoded paths, please modify according to actual situation before use
   - 特别是 `intensity.py` 中的 `main_folders` 和 `earthquake_analysis_combined.py` 中的输出目录
   - Especially `main_folders` in `intensity.py` and output directory in `earthquake_analysis_combined.py`

2. **文件格式** / **File Formats**
   - 确保输入文件格式符合要求（列名、数据类型等）
   - Ensure input file formats meet requirements (column names, data types, etc.)
   - Excel文件应使用 `.xlsx` 格式
   - Excel files should use `.xlsx` format

3. **数据质量** / **Data Quality**
   - 检查输入数据的完整性和准确性
   - Check completeness and accuracy of input data
   - 确保经纬度坐标在合理范围内
   - Ensure latitude/longitude coordinates are within reasonable range

4. **性能考虑** / **Performance Considerations**
   - 大量站点计算可能需要较长时间
   - Large number of site calculations may take considerable time
   - 使用分组功能可以并行处理以提高效率
   - Use grouping function to process in parallel for improved efficiency

---

## 联系信息 / Contact Information

**作者 / Author:** Chaohui.Feng  
**邮箱 / Email:** fengchaohui23@mails.ucas.ac.cn  
**日期 / Date:** 2025.10.01

---

## 许可证 / License

本项目仅供学术研究使用。
This project is for academic research purposes only.

