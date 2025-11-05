import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

# 设置字体：英文优先使用 Times New Roman，中文回退到黑体/雅黑
# Set fonts: English priority Times New Roman, Chinese fallback to SimHei/YaHei
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 烈度顺序和颜色（从高到低）- 包含X级 / Intensity order and colors (high to low) - includes X level
# 修改为更鲜艳的颜色，避免与模拟烈度的黄色混淆 / Modified to brighter colors to avoid confusion with simulated intensity yellow
intensity_order = ["X", "IX", "VIII", "V，II", "VI", "V"]
intensity_colors = {
    "X": "darkred",           # 深红色
    "IX": "crimson",          # 深粉红色
    "VIII": "darkgoldenrod",  # 暗黄色
    "VII": "darkgreen",       # 深绿色
    "VI": "navy",             # 深蓝色
    "V": "darkgray"           # 深灰色
}

# 罗马数字映射 - 包含X级
intensity_str_to_value = {"Ⅹ": 10, "Ⅸ": 9, "Ⅷ": 8, "Ⅶ": 7, "Ⅵ": 6, "Ⅴ": 5}
intensity_order_roman = ["Ⅹ", "Ⅸ", "Ⅷ", "Ⅶ", "Ⅵ", "Ⅴ"]

# 罗马数字历史等震线的独立颜色映射（避免与模拟烈度混淆）
intensity_colors_roman = {
    "Ⅹ": "darkred",           # 深红色
    "Ⅸ": "crimson",          # 深粉红色
    "Ⅷ": "#CDCD00",          # 自定义黄色 RGB(205,205,0)
    "Ⅶ": "darkgreen",        # 深绿色
    "Ⅵ": "navy",             # 深蓝色
    "Ⅴ": "darkgray"          # 深灰色
}

# 烈度分级边界（包含X级）
INT_LEVELS = [0, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

def round_intensity_to_roman(intensity_value):
    """
    将烈度值映射到罗马数字烈度等级 / Map intensity value to Roman numeral intensity level
    包含X级 / Includes X level:
    ≤5.5: V, 5.5-6.5: VI, 6.5-7.5: VII, 7.5-8.5: VIII, 8.5-9.5: IX, >9.5: X
    """
    if intensity_value <= 5.5:
        return 5  # V
    elif intensity_value <= 6.5:
        return 6  # VI
    elif intensity_value <= 7.5:
        return 7  # VII
    elif intensity_value <= 8.5:
        return 8  # VIII
    elif intensity_value <= 9.2:
        return 9  # IX
    else:
        return 10  # X

def get_intensity_color(intensity_value, cmap, norm):
    """
    根据四舍五入的烈度值获取颜色 / Get color based on rounded intensity value
    """
    rounded_intensity = round_intensity_to_roman(intensity_value)
    return cmap(norm(rounded_intensity))

def map_intensity_label_to_value(intensity_label):
    """
    将历史等震线中的烈度标注映射为数值 5-10, 支持阿拉伯数字、ASCII 罗马数字和 Unicode 罗马数字
    Map intensity labels in historical isoseismals to values 5-10, supports Arabic numerals, ASCII Roman numerals and Unicode Roman numerals
    """
    try:
        if isinstance(intensity_label, (int, float)):
            return int(round(float(intensity_label)))
        label_str = str(intensity_label).strip()
        label_str = label_str.replace('，', ',').split(',')[0].strip()
        label_upper = label_str.upper()
        ascii_roman_map = {"V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
        if label_upper in ascii_roman_map:
            return ascii_roman_map[label_upper]
        if label_str in intensity_str_to_value:
            return intensity_str_to_value[label_str]
        return int(round(float(label_str)))
    except Exception:
        return None

def get_historical_color(intensity_label, cmap, norm):
    """
    使用与模拟烈度一致的 cmap/norm 为历史等震线选择颜色。
    """
    value = map_intensity_label_to_value(intensity_label)
    if value is None:
        return 'black'
    value = max(5, min(10, value))
    # 覆盖 VIII 的颜色为 RGB(205,205,0)
    if int(round(value)) == 8:
        return '#CDCD00'
    return cmap(norm(value))

def value_to_roman(value):
    mapping = {5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'}
    return mapping.get(int(value), str(value))

def format_intensity_label_for_display(intensity_label):
    """
    将各种格式的历史烈度标注转换为标准罗马数字字符串用于标注显示。
    """
    value = map_intensity_label_to_value(intensity_label)
    if value is None:
        return str(intensity_label)
    value = max(5, min(10, int(round(value))))
    return value_to_roman(value)

def smooth_curve(x, y, smooth_factor=100):
    """
    history_line.py的平滑曲线函数 / Smooth curve function from history_line.py
    """
    try:
        tck, _ = splprep([x, y], s=0)
        new_points = splev(np.linspace(0, 1, smooth_factor), tck)
        return new_points
    except:
        return x, y

def smooth_curve_periodic(x, y, smooth_factor=100):
    """
    result_history.py的周期性平滑曲线函数 / Periodic smooth curve function from result_history.py
    """
    try:
        tck, u = splprep([x, y], s=0, per=True)
        u_new = np.linspace(u.min(), u.max(), smooth_factor)
        x_new, y_new = splev(u_new, tck, der=0)
        return x_new, y_new
    except:
        return x, y

class EarthquakeAnalysisGUI:
    """
    地震烈度综合分析工具GUI类 / Comprehensive Earthquake Intensity Analysis Tool GUI Class
    """
    def __init__(self):
        """
        初始化GUI界面 / Initialize GUI interface
        """
        self.root = tk.Tk()
        self.root.title("地震烈度综合分析工具")
        self.root.geometry("600x500")
        
        # 文件路径变量
        self.line_path_var = tk.StringVar()
        self.point_path_var = tk.StringVar()
        self.fault_path_var = tk.StringVar()  # 新增：断层数据文件路径
        self.sim_paths_var = tk.StringVar()
        self.output_name_var = tk.StringVar()
        self.title_var = tk.StringVar()
        # 图例位置：'lower left' 或 'upper right'
        self.legend_loc_var = tk.StringVar(value="右上角")
        # 可选：绘图范围（经纬度裁剪）
        self.use_bbox_var = tk.BooleanVar(value=False)
        self.min_lon_var = tk.DoubleVar()
        self.max_lon_var = tk.DoubleVar()
        self.min_lat_var = tk.DoubleVar()
        self.max_lat_var = tk.DoubleVar()
        self.bbox = None  # (min_lon, max_lon, min_lat, max_lat) 或 None
        
        # 数据存储
        self.line_df = None
        self.point_df = None
        self.fault_df = None  # 新增：断层数据存储
        
        self.setup_gui()
        
    def setup_gui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 标题
        title_label = tk.Label(main_frame, text="地震烈度综合分析工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 第一步：历史等震线数据
        step1_frame = ttk.LabelFrame(main_frame, text="第一步：历史等震线数据", padding=10)
        step1_frame.pack(fill='x', pady=10)
        
        # 等震线数据文件选择
        tk.Label(step1_frame, text="等震线数据文件:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(step1_frame, textvariable=self.line_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(step1_frame, text="选择文件", command=self.select_line_file, bg="lightblue").grid(row=0, column=2, padx=5, pady=5)
        
        # 城市点数据文件选择
        tk.Label(step1_frame, text="城市点数据文件:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(step1_frame, textvariable=self.point_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(step1_frame, text="选择文件", command=self.select_point_file, bg="lightblue").grid(row=1, column=2, padx=5, pady=5)
        
        # 断层数据文件选择
        tk.Label(step1_frame, text="断层数据文件:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(step1_frame, textvariable=self.fault_path_var, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(step1_frame, text="选择文件", command=self.select_fault_file, bg="lightblue").grid(row=2, column=2, padx=5, pady=5)
        
        # 第二步：模拟数据
        step2_frame = ttk.LabelFrame(main_frame, text="第二步：模拟数据", padding=10)
        step2_frame.pack(fill='x', pady=10)
        
        # 模拟数据文件选择
        tk.Label(step2_frame, text="模拟CSV文件:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(step2_frame, textvariable=self.sim_paths_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(step2_frame, text="选择文件", command=self.select_sim_files, bg="lightgreen").grid(row=0, column=2, padx=5, pady=5)
        
        # 输出设置
        output_frame = ttk.LabelFrame(main_frame, text="输出设置", padding=10)
        output_frame.pack(fill='x', pady=10)
        
        tk.Label(output_frame, text="图片标题:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(output_frame, textvariable=self.title_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(output_frame, text="输出文件名:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(output_frame, textvariable=self.output_name_var, width=50).grid(row=1, column=1, padx=5, pady=5)

        # 图例位置选择
        tk.Label(output_frame, text="图例位置:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        legend_combo = ttk.Combobox(output_frame, textvariable=self.legend_loc_var, state="readonly", width=20)
        legend_combo['values'] = ("左下角", "右上角")
        legend_combo.current(1)
        legend_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # 可选：绘图经纬度范围设置
        bbox_frame = ttk.LabelFrame(main_frame, text="可选：绘图经纬度范围（启用后仅在该范围内作图）", padding=10)
        bbox_frame.pack(fill='x', pady=10)
        tk.Checkbutton(bbox_frame, text="启用范围裁剪", variable=self.use_bbox_var).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Label(bbox_frame, text="经度最小:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        tk.Spinbox(bbox_frame, textvariable=self.min_lon_var, width=12, from_=-180.0, to=180.0, increment=0.01, format='%.2f').grid(row=1, column=1, padx=5, pady=2)
        tk.Label(bbox_frame, text="经度最大:").grid(row=1, column=2, sticky="e", padx=5, pady=2)
        tk.Spinbox(bbox_frame, textvariable=self.max_lon_var, width=12, from_=-180.0, to=180.0, increment=0.01, format='%.2f').grid(row=1, column=3, padx=5, pady=2)
        tk.Label(bbox_frame, text="纬度最小:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        tk.Spinbox(bbox_frame, textvariable=self.min_lat_var, width=12, from_=-90.0, to=90.0, increment=0.01, format='%.2f').grid(row=2, column=1, padx=5, pady=2)
        tk.Label(bbox_frame, text="纬度最大:").grid(row=2, column=2, sticky="e", padx=5, pady=2)
        tk.Spinbox(bbox_frame, textvariable=self.max_lat_var, width=12, from_=-90.0, to=90.0, increment=0.01, format='%.2f').grid(row=2, column=3, padx=5, pady=2)
        
        # 生成按钮
        generate_button = tk.Button(main_frame, text="生成综合分析图", command=self.generate_analysis, 
                                    bg="orange", font=("Arial", 14, "bold"), height=2)
        generate_button.pack(pady=20)
        
        # 说明文字
        info_label = tk.Label(main_frame, text="说明：\n1. 先选择历史等震线数据、城市点数据和断层数据\n2. 再选择模拟CSV文件（可多选）\n3. 输入图片标题和输出文件名\n4. 点击生成按钮创建综合分析图", 
                              font=("Arial", 9), fg="gray", justify="left")
        info_label.pack(pady=10)
        
    def select_line_file(self):
        filename = filedialog.askopenfilename(title="选择历史等震线Excel文件", filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.line_path_var.set(filename)
            
    def select_point_file(self):
        filename = filedialog.askopenfilename(title="选择城市点数据Excel文件", filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.point_path_var.set(filename)
            
    def select_fault_file(self):
        filename = filedialog.askopenfilename(title="选择断层数据Excel文件", filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.fault_path_var.set(filename)
            
    def select_sim_files(self):
        filenames = filedialog.askopenfilenames(title="选择一个或多个模拟CSV文件", filetypes=[("CSV files", "*.csv")])
        if filenames:
            self.sim_paths_var.set("; ".join(filenames))
            
    def generate_analysis(self):
        """
        生成综合分析图 / Generate comprehensive analysis plot
        """
        # 检查文件是否都已选择 / Check if all files are selected
        if not self.line_path_var.get() or not self.point_path_var.get() or not self.fault_path_var.get() or not self.sim_paths_var.get():
            messagebox.showerror("错误", "请确保选择了所有需要的文件。 / Please ensure all required files are selected.")
            return
            
        # 检查输出设置
        if not self.title_var.get().strip():
            messagebox.showerror("错误", "请输入图片标题。")
            return
            
        if not self.output_name_var.get().strip():
            messagebox.showerror("错误", "请输入输出文件名。")
            return
            
        try:
            print("开始读取文件...")
            
            # 读取历史等震线数据
            print(f"读取等震线文件: {self.line_path_var.get()}")
            self.line_df = pd.read_excel(self.line_path_var.get())
            print(f"等震线数据形状: {self.line_df.shape}")
            print(f"等震线列名: {list(self.line_df.columns)}")
            print(f"等震线intensity列的唯一值: {self.line_df['intensity'].unique()}")
            
            print(f"读取城市点文件: {self.point_path_var.get()}")
            self.point_df = pd.read_excel(self.point_path_var.get())
            print(f"城市点数据形状: {self.point_df.shape}")
            print(f"城市点列名: {list(self.point_df.columns)}")
            
            # 读取断层数据
            print(f"读取断层文件: {self.fault_path_var.get()}")
            self.fault_df = pd.read_excel(self.fault_path_var.get())
            print(f"断层数据形状: {self.fault_df.shape}")
            print(f"断层列名: {list(self.fault_df.columns)}")
            
            # 读取模拟数据
            sim_paths = self.sim_paths_var.get().split("; ")
            print(f"模拟文件数量: {len(sim_paths)}")
            
            all_lons = []
            all_lats = []
            all_intensities = []
            
            for i, sim_path in enumerate(sim_paths):
                print(f"读取模拟文件 {i+1}: {sim_path}")
                sim_df = pd.read_csv(sim_path)
                print(f"文件 {i+1} 数据形状: {sim_df.shape}")
                print(f"文件 {i+1} 列名: {list(sim_df.columns)}")
                
                # 检查必要的列是否存在
                if "Longitude" not in sim_df.columns or "Latitude" not in sim_df.columns or "Intensity" not in sim_df.columns:
                    raise ValueError(f"模拟文件 {sim_path} 缺少必要的列 (Longitude, Latitude, Intensity)")
                
                all_lons.extend(sim_df["Longitude"].tolist())
                all_lats.extend(sim_df["Latitude"].tolist())
                all_intensities.extend(sim_df["Intensity"].tolist())
            
            # 应用可选的经纬度范围裁剪
            if self.use_bbox_var.get():
                try:
                    min_lon = float(self.min_lon_var.get())
                    max_lon = float(self.max_lon_var.get())
                    min_lat = float(self.min_lat_var.get())
                    max_lat = float(self.max_lat_var.get())
                    if not (min_lon < max_lon and min_lat < max_lat):
                        raise ValueError("经纬度范围无效：最小值必须小于最大值")
                    self.bbox = (min_lon, max_lon, min_lat, max_lat)
                    print(f"启用范围裁剪: {self.bbox}")
                except Exception as bbox_err:
                    messagebox.showerror("错误", f"经纬度范围输入无效：{bbox_err}")
                    return
            else:
                self.bbox = None
            
            print(f"总数据点数: {len(all_lons)}")
            print(f"经度范围: {min(all_lons):.3f} - {max(all_lons):.3f}")
            print(f"纬度范围: {min(all_lats):.3f} - {max(all_lats):.3f}")
            print(f"烈度范围: {min(all_intensities):.3f} - {max(all_intensities):.3f}")
            
            # 创建综合分析图
            self.create_combined_plot(all_lons, all_lats, all_intensities)
            
        except Exception as e:
            error_msg = f"生成图像时出错：{str(e)}\n\n详细错误信息：{type(e).__name__}"
            print(f"错误: {error_msg}")
            messagebox.showerror("错误", error_msg)
            
    def create_combined_plot(self, all_lons, all_lats, all_intensities):
        """
        创建综合分析图 / Create comprehensive analysis plot
        
        参数 / Parameters:
            all_lons: 所有经度数据 / All longitude data
            all_lats: 所有纬度数据 / All latitude data
            all_intensities: 所有烈度数据 / All intensity data
        """
        try:
            print("开始创建图像... / Starting to create image...")
            
            # 创建图像
            plt.figure(figsize=(12, 10))
            
            # 计算绘图范围：使用用户裁剪框或数据整体范围
            all_lon_min, all_lon_max = min(all_lons), max(all_lons)
            all_lat_min, all_lat_max = min(all_lats), max(all_lats)
            if self.bbox is not None:
                bbox_min_lon, bbox_max_lon, bbox_min_lat, bbox_max_lat = self.bbox
                # 先对输入数据进行裁剪
                mask_points = [
                    (bbox_min_lon <= lo <= bbox_max_lon) and (bbox_min_lat <= la <= bbox_max_lat)
                    for lo, la in zip(all_lons, all_lats)
                ]
                # 若裁剪后没有数据，提示并返回
                if not any(mask_points):
                    messagebox.showerror("错误", "范围裁剪后没有可用的模拟数据点。")
                    return
                all_lons = [lo for lo, keep in zip(all_lons, mask_points) if keep]
                all_lats = [la for la, keep in zip(all_lats, mask_points) if keep]
                all_intensities = [ia for ia, keep in zip(all_intensities, mask_points) if keep]
                all_lon_min, all_lon_max = bbox_min_lon, bbox_max_lon
                all_lat_min, all_lat_max = bbox_min_lat, bbox_max_lat
            
            print(f"数据范围 - 经度: {all_lon_min:.3f} - {all_lon_max:.3f}, 纬度: {all_lat_min:.3f} - {all_lat_max:.3f}")
            
            # 使用result_history.py的网格创建方法
            xi = np.linspace(all_lon_min, all_lon_max, 400)
            yi = np.linspace(all_lat_min, all_lat_max, 400)
            xi, yi = np.meshgrid(xi, yi)
            
            print(f"网格大小: {xi.shape}")
            
            # 使用温和凸包插值法防止内凹趋势
            print("进行温和凸包插值...")
            
            # 计算所有数据点的凸包（基于裁剪后的点集）
            all_points = np.column_stack([all_lons, all_lats])
            hull = ConvexHull(all_points)
            hull_points = all_points[hull.vertices]
            
            # 计算数据范围
            lon_range = all_lon_max - all_lon_min
            lat_range = all_lat_max - all_lat_min
            max_range = max(lon_range, lat_range)
            
            # 温和扩展凸包边界，保持适度外凸
            hull_center = np.mean(hull_points, axis=0)
            expanded_hull_points = []
            
            for point in hull_points:
                # 向外适度扩展凸包点，保持外凸效果
                direction = point - hull_center
                distance = np.linalg.norm(direction)
                if distance > 0:
                    # 使用较小的扩展比例，保持适度外凸
                    expansion_factor = 0.05 + (distance / max_range) * 0.1  # 5%-15%的扩展
                    expanded_point = point + direction * expansion_factor
                    expanded_hull_points.append(expanded_point)
                else:
                    expanded_hull_points.append(point)
            
            expanded_hull_points = np.array(expanded_hull_points)
            
            # 创建扩展凸包路径
            expanded_hull_path = Path(expanded_hull_points)
            
            # 在扩展凸包内进行插值
            print("在扩展凸包内进行插值...")
            
            # 创建更密集的网格用于插值
            xi_dense = np.linspace(all_lon_min - lon_range * 0.1, all_lon_max + lon_range * 0.1, 600)
            yi_dense = np.linspace(all_lat_min - lat_range * 0.1, all_lat_max + lat_range * 0.1, 600)
            xi_dense, yi_dense = np.meshgrid(xi_dense, yi_dense)
            
            # 在密集网格上进行插值（仅在裁剪框内）
            zi_dense = griddata((all_lons, all_lats), all_intensities, (xi_dense, yi_dense), method="cubic")
            
            # 应用扩展凸包掩码
            mask_dense = expanded_hull_path.contains_points(np.column_stack([xi_dense.flatten(), yi_dense.flatten()]))
            mask_dense = mask_dense.reshape(xi_dense.shape)
            zi_dense[~mask_dense] = np.nan
            
            # 对凸包内的值进行温和平滑处理
            zi_smooth = gaussian_filter(zi_dense, sigma=1.0)
            zi_dense = np.where(np.isnan(zi_dense), zi_dense, zi_smooth)
            
            # 将密集网格的结果插值回原始网格
            valid_mask = ~np.isnan(zi_dense)
            if np.any(valid_mask):
                zi = griddata((xi_dense[valid_mask], yi_dense[valid_mask]), 
                             zi_dense[valid_mask], (xi, yi), method="linear")
            else:
                zi = griddata((all_lons, all_lats), all_intensities, (xi, yi), method="cubic")
            
            # 最终应用掩码到原始网格：扩展凸包与可选裁剪框
            mask = expanded_hull_path.contains_points(np.column_stack([xi.flatten(), yi.flatten()]))
            mask = mask.reshape(xi.shape)
            if self.bbox is not None:
                bbox_mask = (xi >= all_lon_min) & (xi <= all_lon_max) & (yi >= all_lat_min) & (yi <= all_lat_max)
                mask = mask & bbox_mask
            zi[~mask] = np.nan
            
            # 额外的外凸处理：确保等值线向外凸出
            print("应用额外外凸处理...")
            
            # 计算每个网格点到凸包中心的距离
            center_lon = np.mean(all_lons)
            center_lat = np.mean(all_lats)
            
            # 对每个网格点进行外凸调整
            for i in range(xi.shape[0]):
                for j in range(xi.shape[1]):
                    if mask[i, j] and not np.isnan(zi[i, j]):
                        # 计算当前点到中心的距离
                        point_lon, point_lat = xi[i, j], yi[i, j]
                        distance_to_center = np.sqrt((point_lon - center_lon)**2 + (point_lat - center_lat)**2)
                        
                        # 计算该点到最近凸包边界的距离
                        min_distance_to_hull = float('inf')
                        for k in range(len(expanded_hull_points) - 1):
                            p1 = expanded_hull_points[k]
                            p2 = expanded_hull_points[k + 1]
                            
                            # 计算点到线段的距离
                            line_vec = p2 - p1
                            point_vec = np.array([point_lon, point_lat]) - p1
                            line_len = np.linalg.norm(line_vec)
                            
                            if line_len > 0:
                                t = max(0, min(1, np.dot(point_vec, line_vec) / (line_len**2)))
                                closest_point = p1 + t * line_vec
                                dist = np.linalg.norm(np.array([point_lon, point_lat]) - closest_point)
                                min_distance_to_hull = min(min_distance_to_hull, dist)
                        
                        # 固定系数为1：不对烈度值做外凸放大
                        if min_distance_to_hull < float('inf'):
                            convex_factor = 1.0
                            zi[i, j] = zi[i, j] * convex_factor
            
            # 使用烈度分级颜色映射（修复四舍五入bug）
            bounds = [0, 5.5, 6.5, 7.5, 8.5, 9.2, 10.5]        # Ⅴ↓, Ⅵ, Ⅶ, Ⅷ, Ⅸ, Ⅹ
            colors = ["white", "blue", "green", "yellow",
                      "orange", "red"]                         # 对应 V↓→Ⅹ
            cmap   = ListedColormap(colors)
            norm   = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
            
            # 对插值结果进行四舍五入处理
            print("应用四舍五入烈度分级...")
            zi_rounded = np.copy(zi)
            for i in range(zi.shape[0]):
                for j in range(zi.shape[1]):
                    if not np.isnan(zi[i, j]):
                        # 智能限制烈度值上限
                        # 如果值大于9.5，检查是否可能是外凸处理导致的
                        # 这里我们采用一个更保守的方法：只有当值超过10.0时才认为是外凸处理导致的
                        if zi[i, j] > 10.0:
                            limited_value = 9.5  # 限制为9.5，避免过度外凸
                        else:
                            limited_value = zi[i, j]  # 保持原值
                        zi_rounded[i, j] = round_intensity_to_roman(limited_value)
            
            # 绘制模拟烈度等值线（包含X级）
            print("绘制模拟烈度等值线...")
            # 使用映射后的数据进行填充
            contourf = plt.contourf(xi, yi, zi_rounded, levels=np.arange(5, 10.5, 0.5), cmap=cmap, norm=norm, alpha=0.6)
            # 使用映射后的数据进行等值线绘制
            contour = plt.contour(xi, yi, zi_rounded, levels=np.arange(5, 10.5, 1), colors='gray', linewidths=0.5)
            
            # 自定义等值线标签格式，将阿拉伯数字转换为罗马数字
            def roman_label_formatter(x):
                if x == 5:
                    return 'V'
                elif x == 6:
                    return 'VI'
                elif x == 7:
                    return 'VII'
                elif x == 8:
                    return 'VIII'
                elif x == 9:
                    return 'IX'
                elif x == 10:
                    return 'X'
                else:
                    return str(int(x))
            
            plt.clabel(contour, inline=True, fontsize=17, fmt=roman_label_formatter, colors='k')
            
            # 绘制历史等震线（对城市点与历史线也进行范围裁剪）
            print("绘制历史等震线...")
            print(f"历史等震线数据形状: {self.line_df.shape}")
            print(f"历史等震线列名: {list(self.line_df.columns)}")
            print(f"历史等震线intensity列的唯一值: {self.line_df['intensity'].unique()}")
            
            # 详细分析每个烈度等级的数据
            print("=== 详细分析每个烈度等级 ===")
            for intensity in self.line_df['intensity'].unique():
                group = self.line_df[self.line_df["intensity"] == intensity]
                print(f"烈度 {intensity}: {len(group)} 个数据点")
                if len(group) > 0:
                    print(f"  经度范围: {group['Longtitude'].min():.3f} - {group['Longtitude'].max():.3f}")
                    print(f"  纬度范围: {group['Latitude'].min():.3f} - {group['Latitude'].max():.3f}")
            print("=== 分析完成 ===")
            
            # 存储已绘制的等震线路径，用于确保不交叉
            drawn_paths = []
            
            # 首先尝试阿拉伯数字格式（history_line.py的格式）
            print("=== 尝试阿拉伯数字格式 ===")
            for intensity in intensity_order:
                group = self.line_df[self.line_df["intensity"] == intensity]
                if self.bbox is not None:
                    group = group[(group["Longtitude"] >= all_lon_min) & (group["Longtitude"] <= all_lon_max) &
                                  (group["Latitude"] >= all_lat_min) & (group["Latitude"] <= all_lat_max)]
                print(f"烈度 {intensity} 数据点数: {len(group)}")
                if len(group) >= 2:  # 降低数据点要求从3个到2个
                    points = group[["Longtitude", "Latitude"]].values
                    try:
                        # 尝试使用凸包插值法
                        if len(points) >= 3:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            
                            # 计算等震线数据范围
                            lon_range = np.max(points[:, 0]) - np.min(points[:, 0])
                            lat_range = np.max(points[:, 1]) - np.min(points[:, 1])
                            max_range = max(lon_range, lat_range)
                            
                            # 温和扩展历史等震线凸包，保持适度外凸
                            hull_center = np.mean(hull_points, axis=0)
                            expanded_hull_points = []
                            for point in hull_points:
                                direction = point - hull_center
                                distance = np.linalg.norm(direction)
                                if distance > 0:
                                    # 使用较小的扩展比例，保持适度外凸
                                    expansion_factor = 0.05 + (distance / max_range) * 0.15  # 5%-20%的扩展
                                    expanded_point = point + direction * expansion_factor
                                    expanded_hull_points.append(expanded_point)
                                else:
                                    expanded_hull_points.append(point)
                            
                            expanded_hull_points = np.array(expanded_hull_points)
                            expanded_hull_points = np.concatenate([expanded_hull_points, expanded_hull_points[:1]])
                            smooth_lon, smooth_lat = smooth_curve(expanded_hull_points[:, 0], expanded_hull_points[:, 1])
                        else:
                            # 如果点数不足，直接使用原始点进行平滑
                            points_closed = np.concatenate([points, points[:1]])
                            smooth_lon, smooth_lat = smooth_curve(points_closed[:, 0], points_closed[:, 1])
                        
                        # 检查是否与已绘制的等震线交叉（放宽检查条件）
                        current_path = Path(np.column_stack([smooth_lon, smooth_lat]))
                        is_valid = True
                        
                        # 只检查与相邻烈度等震线的交叉，而不是所有等震线
                        for prev_path in drawn_paths[-2:]:  # 只检查最近绘制的2条等震线
                            try:
                                # 检查是否有明显的交叉
                                if prev_path.intersects_path(current_path):
                                    # 如果交叉，检查是否是合理的嵌套关系
                                    if not (prev_path.contains_path(current_path) or current_path.contains_path(prev_path)):
                                        is_valid = False
                                        print(f"烈度 {intensity} 与之前的等震线存在不合理交叉，跳过")
                                        break
                            except Exception as cross_error:
                                print(f"交叉检查出错，继续绘制: {cross_error}")
                                continue
                        
                        if is_valid:
                            plt.plot(smooth_lon, smooth_lat, color=get_historical_color(intensity, cmap, norm),
                                     label=f"Historical Intensity {intensity}", linewidth=2)
                            drawn_paths.append(current_path)
                            print(f"成功绘制烈度 {intensity} 等震线")
                    except Exception as e:
                        print(f"绘制烈度 {intensity} 等震线时出错: {e}")
                        # 尝试直接绘制原始点
                        try:
                            if len(points) >= 2:
                                plt.plot(points[:, 0], points[:, 1], color=get_historical_color(intensity, cmap, norm),
                                         label=f"Historical Intensity {intensity} (raw)", linewidth=1, linestyle='--')
                                print(f"使用原始点绘制烈度 {intensity} 等震线")
                        except Exception as fallback_error:
                            print(f"原始点绘制也失败: {fallback_error}")
                        continue
            
            # 如果没有找到等震线，尝试罗马数字格式（result_history.py的格式）
            if len(drawn_paths) == 0:
                print("=== 尝试罗马数字格式 ===")
                for intensity in intensity_order_roman:
                    group = self.line_df[self.line_df["intensity"] == intensity]
                    if self.bbox is not None:
                        group = group[(group["Longtitude"] >= all_lon_min) & (group["Longtitude"] <= all_lon_max) &
                                      (group["Latitude"] >= all_lat_min) & (group["Latitude"] <= all_lat_max)]
                    print(f"烈度 {intensity} 数据点数: {len(group)}")
                    if len(group) >= 2:  # 降低数据点要求
                        points = group[["Longtitude", "Latitude"]].values
                        try:
                            if len(points) >= 3:
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                hull_points = np.concatenate([hull_points, hull_points[:1]])
                                smooth_lon, smooth_lat = smooth_curve_periodic(hull_points[:, 0], hull_points[:, 1])
                            else:
                                # 如果点数不足，直接使用原始点
                                points_closed = np.concatenate([points, points[:1]])
                                smooth_lon, smooth_lat = smooth_curve_periodic(points_closed[:, 0], points_closed[:, 1])
                            
                            # 使用与模拟烈度一致的颜色
                            color = get_historical_color(intensity, cmap, norm)
                            plt.plot(smooth_lon, smooth_lat, color=color,
                                     label=f"Historical Intensity {intensity}", linewidth=2)
                            drawn_paths.append(Path(np.column_stack([smooth_lon, smooth_lat])))
                            print(f"成功绘制烈度 {intensity} 等震线")
                        except Exception as e:
                            print(f"绘制烈度 {intensity} 等震线时出错: {e}")
                            # 尝试直接绘制原始点
                            try:
                                if len(points) >= 2:
                                    # 使用与模拟烈度一致的颜色
                                    color = get_historical_color(intensity, cmap, norm)
                                    plt.plot(points[:, 0], points[:, 1], color=color,
                                             label=f"Historical Intensity {intensity} (raw)", linewidth=1, linestyle='--')
                                    print(f"使用原始点绘制烈度 {intensity} 等震线")
                            except Exception as fallback_error:
                                print(f"原始点绘制也失败: {fallback_error}")
                            continue
            
            # 如果仍然没有找到等震线，尝试直接绘制所有可能的格式
            if len(drawn_paths) == 0:
                print("=== 尝试直接绘制所有格式 ===")
                # 获取所有唯一的烈度值
                all_intensities = self.line_df['intensity'].unique()
                print(f"所有烈度值: {all_intensities}")
                
                for intensity in all_intensities:
                    group = self.line_df[self.line_df["intensity"] == intensity]
                    print(f"烈度 {intensity} 数据点数: {len(group)}")
                    if len(group) >= 2:  # 降低数据点要求
                        points = group[["Longtitude", "Latitude"]].values
                        try:
                            # 尝试使用凸包插值法
                            if len(points) >= 3:
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                
                                # 计算等震线数据范围
                                lon_range = np.max(points[:, 0]) - np.min(points[:, 0])
                                lat_range = np.max(points[:, 1]) - np.min(points[:, 1])
                                max_range = max(lon_range, lat_range)
                                
                                # 温和扩展历史等震线凸包，保持适度外凸
                                hull_center = np.mean(hull_points, axis=0)
                                expanded_hull_points = []
                                for point in hull_points:
                                    direction = point - hull_center
                                    distance = np.linalg.norm(direction)
                                    if distance > 0:
                                        # 使用较小的扩展比例，保持适度外凸
                                        expansion_factor = 0.05 + (distance / max_range) * 0.15  # 5%-20%的扩展
                                        expanded_point = point + direction * expansion_factor
                                        expanded_hull_points.append(expanded_point)
                                    else:
                                        expanded_hull_points.append(point)
                                
                                expanded_hull_points = np.array(expanded_hull_points)
                                expanded_hull_points = np.concatenate([expanded_hull_points, expanded_hull_points[:1]])
                                smooth_lon, smooth_lat = smooth_curve(expanded_hull_points[:, 0], expanded_hull_points[:, 1])
                            else:
                                # 如果点数不足，直接使用原始点进行平滑
                                points_closed = np.concatenate([points, points[:1]])
                                smooth_lon, smooth_lat = smooth_curve(points_closed[:, 0], points_closed[:, 1])
                            
                            # 历史等震线统一使用黑色, 并标注烈度
                            color = 'black'
                            
                            plt.plot(smooth_lon, smooth_lat, color=color,
                                     label=f"Historical Intensity {intensity}", linewidth=2)
                            mid_idx = len(smooth_lon) // 2
                            mid_x, mid_y = smooth_lon[mid_idx], smooth_lat[mid_idx]
                            plt.text(mid_x, mid_y, format_intensity_label_for_display(intensity),
                                     fontsize=14, color='black', ha='center', va='center',
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))
                            drawn_paths.append(Path(np.column_stack([smooth_lon, smooth_lat])))
                            print(f"成功绘制烈度 {intensity} 等震线")
                        except Exception as e:
                            print(f"绘制烈度 {intensity} 等震线时出错: {e}")
                            # 尝试直接绘制原始点
                            try:
                                if len(points) >= 2:
                                    color = get_historical_color(intensity, cmap, norm)
                                    
                                    plt.plot(points[:, 0], points[:, 1], color=color,
                                             label=f"Historical Intensity {intensity} (raw)", linewidth=1, linestyle='--')
                                    print(f"使用原始点绘制烈度 {intensity} 等震线")
                            except Exception as fallback_error:
                                print(f"原始点绘制也失败: {fallback_error}")
                            continue
            
            # 如果交叉检查导致等震线太少，尝试简化版本（不进行交叉检查）
            if len(drawn_paths) < 2:
                print("=== 交叉检查过于严格，尝试简化版本 ===")
                drawn_paths = []  # 清空之前的路径
                
                # 首先尝试所有可能的烈度格式，不进行交叉检查
                print("=== 尝试所有烈度格式（无交叉检查） ===")
                all_possible_intensities = list(intensity_order) + list(intensity_order_roman) + list(self.line_df['intensity'].unique())
                all_possible_intensities = list(dict.fromkeys(all_possible_intensities))  # 去重
                print(f"所有可能的烈度值: {all_possible_intensities}")
                
                for intensity in all_possible_intensities:
                    group = self.line_df[self.line_df["intensity"] == intensity]
                    print(f"简化版本 - 烈度 {intensity} 数据点数: {len(group)}")
                    if len(group) >= 1:  # 只要有数据点就尝试绘制
                        points = group[["Longtitude", "Latitude"]].values
                        try:
                            # 尝试使用凸包插值法
                            if len(points) >= 3:
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                
                                # 计算等震线数据范围
                                lon_range = np.max(points[:, 0]) - np.min(points[:, 0])
                                lat_range = np.max(points[:, 1]) - np.min(points[:, 1])
                                max_range = max(lon_range, lat_range)
                                
                                # 温和扩展历史等震线凸包，保持适度外凸
                                hull_center = np.mean(hull_points, axis=0)
                                expanded_hull_points = []
                                for point in hull_points:
                                    direction = point - hull_center
                                    distance = np.linalg.norm(direction)
                                    if distance > 0:
                                        # 使用较小的扩展比例，保持适度外凸
                                        expansion_factor = 0.05 + (distance / max_range) * 0.15  # 5%-20%的扩展
                                        expanded_point = point + direction * expansion_factor
                                        expanded_hull_points.append(expanded_point)
                                    else:
                                        expanded_hull_points.append(point)
                                
                                expanded_hull_points = np.array(expanded_hull_points)
                                expanded_hull_points = np.concatenate([expanded_hull_points, expanded_hull_points[:1]])
                                smooth_lon, smooth_lat = smooth_curve(expanded_hull_points[:, 0], expanded_hull_points[:, 1])
                            else:
                                # 如果点数不足，直接使用原始点进行平滑
                                points_closed = np.concatenate([points, points[:1]])
                                smooth_lon, smooth_lat = smooth_curve(points_closed[:, 0], points_closed[:, 1])
                            
                            # 选择与模拟烈度一致的颜色
                            color = get_historical_color(intensity, cmap, norm)
                            
                            plt.plot(smooth_lon, smooth_lat, color=color,
                                     label=f"历史烈度 {intensity}", linewidth=2)
                            drawn_paths.append(Path(np.column_stack([smooth_lon, smooth_lat])))
                            print(f"简化版本 - 成功绘制烈度 {intensity} 等震线")
                        except Exception as e:
                            print(f"简化版本 - 绘制烈度 {intensity} 等震线时出错: {e}")
                            # 尝试直接绘制原始点
                            try:
                                if len(points) >= 1:
                                    color = 'black'
                                    
                                    plt.plot(points[:, 0], points[:, 1], color=color,
                                             label=f"历史烈度 {intensity} (原始)", linewidth=1, linestyle='--')
                                    mid_idx = len(points) // 2
                                    mid_x, mid_y = points[mid_idx, 0], points[mid_idx, 1]
                                    plt.text(mid_x, mid_y, format_intensity_label_for_display(intensity),
                                             fontsize=14, color='black', ha='center', va='center',
                                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))
                                    print(f"简化版本 - 使用原始点绘制烈度 {intensity} 等震线")
                            except Exception as fallback_error:
                                print(f"简化版本 - 原始点绘制也失败: {fallback_error}")
                            continue
                
                # 重新尝试阿拉伯数字格式，但不进行交叉检查
                for intensity in intensity_order:
                    group = self.line_df[self.line_df["intensity"] == intensity]
                    print(f"简化版本 - 烈度 {intensity} 数据点数: {len(group)}")
                    if len(group) >= 2:  # 降低数据点要求
                        points = group[["Longtitude", "Latitude"]].values
                        try:
                            # 尝试使用凸包插值法
                            if len(points) >= 3:
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                
                                # 计算等震线数据范围
                                lon_range = np.max(points[:, 0]) - np.min(points[:, 0])
                                lat_range = np.max(points[:, 1]) - np.min(points[:, 1])
                                max_range = max(lon_range, lat_range)
                                
                                # 温和扩展历史等震线凸包，保持适度外凸
                                hull_center = np.mean(hull_points, axis=0)
                                expanded_hull_points = []
                                for point in hull_points:
                                    direction = point - hull_center
                                    distance = np.linalg.norm(direction)
                                    if distance > 0:
                                        # 使用较小的扩展比例，保持适度外凸
                                        expansion_factor = 0.05 + (distance / max_range) * 0.15  # 5%-20%的扩展
                                        expanded_point = point + direction * expansion_factor
                                        expanded_hull_points.append(expanded_point)
                                    else:
                                        expanded_hull_points.append(point)
                                
                                expanded_hull_points = np.array(expanded_hull_points)
                                expanded_hull_points = np.concatenate([expanded_hull_points, expanded_hull_points[:1]])
                                smooth_lon, smooth_lat = smooth_curve(expanded_hull_points[:, 0], expanded_hull_points[:, 1])
                            else:
                                # 如果点数不足，直接使用原始点进行平滑
                                points_closed = np.concatenate([points, points[:1]])
                                smooth_lon, smooth_lat = smooth_curve(points_closed[:, 0], points_closed[:, 1])
                            
                            plt.plot(smooth_lon, smooth_lat, color=get_historical_color(intensity, cmap, norm),
                                     label=f"历史烈度 {intensity}", linewidth=2)
                            drawn_paths.append(Path(np.column_stack([smooth_lon, smooth_lat])))
                            print(f"简化版本 - 成功绘制烈度 {intensity} 等震线")
                        except Exception as e:
                            print(f"简化版本 - 绘制烈度 {intensity} 等震线时出错: {e}")
                            # 尝试直接绘制原始点
                            try:
                                if len(points) >= 2:
                                    plt.plot(points[:, 0], points[:, 1], color=get_historical_color(intensity, cmap, norm),
                                             label=f"历史烈度 {intensity} (原始)", linewidth=1, linestyle='--')
                                    print(f"简化版本 - 使用原始点绘制烈度 {intensity} 等震线")
                            except Exception as fallback_error:
                                print(f"简化版本 - 原始点绘制也失败: {fallback_error}")
                            continue
            
            print(f"成功绘制了 {len(drawn_paths)} 条历史等震线")
            
            # 绘制断层线
            print("绘制断层线...")
            if self.fault_df is not None and len(self.fault_df) > 0:
                try:
                    # 检查断层数据列名
                    if "Longtitude" in self.fault_df.columns and "Latitude" in self.fault_df.columns:
                        # 对断层数据进行范围裁剪
                        fault_df = self.fault_df
                        if self.bbox is not None:
                            fault_df = fault_df[(fault_df["Longtitude"] >= all_lon_min) & (fault_df["Longtitude"] <= all_lon_max) &
                                                (fault_df["Latitude"] >= all_lat_min) & (fault_df["Latitude"] <= all_lat_max)]
                        
                        # 新增：按年份分组绘制的检测与处理
                        possible_year_cols = [col for col in ["year", "Year", "年份", "YR"] if col in fault_df.columns]
                        has_year = len(possible_year_cols) > 0
                        year_drawn = False  # 标记是否成功绘制了年份断层
                        
                        if has_year:
                            year_col = possible_year_cols[0]
                            mark_base = max(all_lon_max - all_lon_min, all_lat_max - all_lat_min)
                            if mark_base <= 0:
                                mark_base = 0.1
                            mark_length = mark_base * 0.02
                            first_label_done = False
                            year_groups_drawn = 0  # 统计成功绘制的年份组数
                            
                            for yvalue, sub in fault_df.groupby(year_col):
                                pts = sub[["Longtitude", "Latitude"]].values
                                if len(pts) < 2:
                                    print(f"年份 {yvalue} 的数据点不足2个，跳过")
                                    continue
                                # 主体虚线
                                plt.plot(pts[:, 0], pts[:, 1], color='darkgray', linestyle='--', linewidth=2,
                                         alpha=0.8, label=("Fault" if not first_label_done else None))
                                first_label_done = True
                                year_groups_drawn += 1
                                
                                # 起点法向
                                s_lon, s_lat = pts[0, 0], pts[0, 1]
                                s_dir_lon = (pts[1, 0] - pts[0, 0])
                                s_dir_lat = (pts[1, 1] - pts[0, 1])
                                s_perp_lon = -s_dir_lat
                                s_perp_lat = s_dir_lon
                                s_len = np.sqrt(s_perp_lon**2 + s_perp_lat**2)
                                if s_len > 0:
                                    s_perp_lon /= s_len
                                    s_perp_lat /= s_len
                                else:
                                    s_perp_lon, s_perp_lat = 0.0, 1.0
                                # 终点法向
                                e_lon, e_lat = pts[-1, 0], pts[-1, 1]
                                e_dir_lon = (pts[-1, 0] - pts[-2, 0])
                                e_dir_lat = (pts[-1, 1] - pts[-2, 1])
                                e_perp_lon = -e_dir_lat
                                e_perp_lat = e_dir_lon
                                e_len = np.sqrt(e_perp_lon**2 + e_perp_lat**2)
                                if e_len > 0:
                                    e_perp_lon /= e_len
                                    e_perp_lat /= e_len
                                else:
                                    e_perp_lon, e_perp_lat = s_perp_lon, s_perp_lat
                                # 画横线首尾
                                plt.plot([s_lon - s_perp_lon * mark_length, s_lon + s_perp_lon * mark_length],
                                         [s_lat - s_perp_lat * mark_length, s_lat + s_perp_lat * mark_length],
                                         color='darkgray', linewidth=3, alpha=0.9, zorder=6)
                                plt.plot([e_lon - e_perp_lon * mark_length, e_lon + e_perp_lon * mark_length],
                                         [e_lat - e_perp_lat * mark_length, e_lat + e_perp_lat * mark_length],
                                         color='darkgray', linewidth=3, alpha=0.9, zorder=6)
                            
                            if year_groups_drawn > 0:
                                year_drawn = True
                                print(f"按年份绘制断层，成功绘制 {year_groups_drawn} 组")
                            else:
                                print("按年份绘制断层失败，所有年份组数据点都不足2个")
                        
                        # 如果没有年份列，或者年份绘制失败，则使用整体绘制
                        if not has_year or not year_drawn:
                            if len(fault_df) >= 2:
                                fault_points = fault_df[["Longtitude", "Latitude"]].values
                                
                                # 使用与历史等震线相同的平滑处理逻辑
                                if len(fault_points) >= 3:
                                    try:
                                        hull = ConvexHull(fault_points)
                                        hull_points = fault_points[hull.vertices]
                                        
                                        # 计算断层数据范围
                                        lon_range = np.max(fault_points[:, 0]) - np.min(fault_points[:, 0])
                                        lat_range = np.max(fault_points[:, 1]) - np.min(fault_points[:, 1])
                                        max_range = max(lon_range, lat_range)
                                        
                                        # 温和扩展断层凸包，保持适度外凸
                                        hull_center = np.mean(hull_points, axis=0)
                                        expanded_hull_points = []
                                        for point in hull_points:
                                            direction = point - hull_center
                                            distance = np.linalg.norm(direction)
                                            if distance > 0:
                                                # 使用较小的扩展比例，保持适度外凸
                                                expansion_factor = 0.05 + (distance / max_range) * 0.15  # 5%-20%的扩展
                                                expanded_point = point + direction * expansion_factor
                                                expanded_hull_points.append(expanded_point)
                                            else:
                                                expanded_hull_points.append(point)
                                        
                                        expanded_hull_points = np.array(expanded_hull_points)
                                        expanded_hull_points = np.concatenate([expanded_hull_points, expanded_hull_points[:1]])
                                        smooth_lon, smooth_lat = smooth_curve(expanded_hull_points[:, 0], expanded_hull_points[:, 1])
                                    except Exception as hull_error:
                                        print(f"断层凸包处理失败，使用原始点: {hull_error}")
                                        # 如果凸包处理失败，直接使用原始点
                                        points_closed = np.concatenate([fault_points, fault_points[:1]])
                                        smooth_lon, smooth_lat = smooth_curve(points_closed[:, 0], points_closed[:, 1])
                                else:
                                    # 如果点数不足，直接使用原始点进行平滑
                                    points_closed = np.concatenate([fault_points, fault_points[:1]])
                                    smooth_lon, smooth_lat = smooth_curve(points_closed[:, 0], points_closed[:, 1])
                            
                                # 绘制断层线（深灰色虚线）
                                plt.plot(smooth_lon, smooth_lat, color='darkgray', linestyle='--', 
                                         linewidth=2, label="Fault", alpha=0.8)
                                
                                # 在断层起始点和结尾点添加竖着的横线标记
                                if len(smooth_lon) > 0 and len(smooth_lat) > 0:
                                     # 起始点（第一个点）
                                     start_lon, start_lat = smooth_lon[0], smooth_lat[0]
                                     # 结尾点（最后一个点）
                                     end_lon, end_lat = smooth_lon[-1], smooth_lat[-1]
                                     
                                     # 计算断层线的方向向量
                                     if len(smooth_lon) > 1:
                                         # 使用前两个点计算起始方向
                                         start_direction_lon = smooth_lon[1] - smooth_lon[0]
                                         start_direction_lat = smooth_lat[1] - smooth_lat[0]
                                         # 使用最后两个点计算结尾方向
                                         end_direction_lon = smooth_lon[-1] - smooth_lon[-2]
                                         end_direction_lat = smooth_lat[-1] - smooth_lat[-2]
                                     else:
                                         # 如果只有一个点，使用默认方向
                                         start_direction_lon, start_direction_lat = 0.01, 0.01
                                         end_direction_lon, end_direction_lat = 0.01, 0.01
                                     
                                     # 计算起始点垂直方向（顺时针旋转90度）
                                     start_perp_lon = -start_direction_lat
                                     start_perp_lat = start_direction_lon
                                     
                                     # 归一化起始点垂直向量
                                     start_perp_length = np.sqrt(start_perp_lon**2 + start_perp_lat**2)
                                     if start_perp_length > 0:
                                         start_perp_lon /= start_perp_length
                                         start_perp_lat /= start_perp_length
                                     else:
                                         # 起点方向退化，使用默认法向量
                                         start_perp_lon, start_perp_lat = 0.0, 1.0
                                     
                                     # 计算结尾点垂直方向（顺时针旋转90度）
                                     end_perp_lon = -end_direction_lat
                                     end_perp_lat = end_direction_lon
                                     
                                     # 归一化结尾点垂直向量
                                     end_perp_length = np.sqrt(end_perp_lon**2 + end_perp_lat**2)
                                     if end_perp_length > 0:
                                         end_perp_lon /= end_perp_length
                                         end_perp_lat /= end_perp_length
                                     else:
                                         # 末端方向退化，则复用起点法向量
                                         end_perp_lon, end_perp_lat = start_perp_lon, start_perp_lat
                                     
                                     # 计算标记线的长度（根据经纬度范围调整）
                                     mark_length = max(lon_range, lat_range) * 0.02  # 2%的范围作为标记长度
                                     
                                     # 绘制起始点标记（竖着的横线）
                                     start_mark_start_lon = start_lon - start_perp_lon * mark_length
                                     start_mark_end_lon = start_lon + start_perp_lon * mark_length
                                     start_mark_start_lat = start_lat - start_perp_lat * mark_length
                                     start_mark_end_lat = start_lat + start_perp_lat * mark_length
                                     
                                     plt.plot([start_mark_start_lon, start_mark_end_lon], [start_mark_start_lat, start_mark_end_lat], 
                                              color='darkgray', linewidth=3, alpha=0.8, zorder=6)
                                     
                                     # 绘制结尾点标记（竖着的横线）
                                     end_mark_start_lon = end_lon - end_perp_lon * mark_length
                                     end_mark_end_lon = end_lon + end_perp_lon * mark_length
                                     end_mark_start_lat = end_lat - end_perp_lat * mark_length
                                     end_mark_end_lat = end_lat + end_perp_lat * mark_length
                                     
                                     plt.plot([end_mark_start_lon, end_mark_end_lon], [end_mark_start_lat, end_mark_end_lat], 
                                              color='darkgray', linewidth=3, alpha=0.8, zorder=6)
                            
                            print(f"成功绘制断层线，数据点数: {len(fault_df)}")
                        else:
                            print("断层数据点数不足，无法绘制")
                    else:
                        print("断层数据缺少必要的列 (Longtitude, Latitude)")
                except Exception as fault_error:
                    print(f"绘制断层线时出错: {fault_error}")
                    # 尝试直接绘制原始点
                    try:
                        if len(self.fault_df) >= 2:
                            fault_points = self.fault_df[["Longtitude", "Latitude"]].values
                            plt.plot(fault_points[:, 0], fault_points[:, 1], color='darkgray', 
                                     linestyle='--', linewidth=1, label="Fault (raw)", alpha=0.6)
                            
                            # 在原始断层起始点和结尾点也添加标记
                            if len(fault_points) > 0:
                                 # 起始点（第一个点）
                                 start_lon, start_lat = fault_points[0, 0], fault_points[0, 1]
                                 # 结尾点（最后一个点）
                                 end_lon, end_lat = fault_points[-1, 0], fault_points[-1, 1]
                                 
                                 # 计算断层线的方向向量
                                 if len(fault_points) > 1:
                                     # 使用前两个点计算起始方向
                                     start_direction_lon = fault_points[1, 0] - fault_points[0, 0]
                                     start_direction_lat = fault_points[1, 1] - fault_points[0, 1]
                                     # 使用最后两个点计算结尾方向
                                     end_direction_lon = fault_points[-1, 0] - fault_points[-2, 0]
                                     end_direction_lat = fault_points[-1, 1] - fault_points[-2, 1]
                                 else:
                                     # 如果只有一个点，使用默认方向
                                     start_direction_lon, start_direction_lat = 0.01, 0.01
                                     end_direction_lon, end_direction_lat = 0.01, 0.01
                                 
                                 # 计算起始点垂直方向（顺时针旋转90度）
                                 start_perp_lon = -start_direction_lat
                                 start_perp_lat = start_direction_lon
                                 
                                 # 归一化起始点垂直向量
                                 start_perp_length = np.sqrt(start_perp_lon**2 + start_perp_lat**2)
                                 if start_perp_length > 0:
                                     start_perp_lon /= start_perp_length
                                     start_perp_lat /= start_perp_length
                                 
                                 # 计算结尾点垂直方向（顺时针旋转90度）
                                 end_perp_lon = -end_direction_lat
                                 end_perp_lat = end_direction_lon
                                 
                                 # 归一化结尾点垂直向量
                                 end_perp_length = np.sqrt(end_perp_lon**2 + end_perp_lat**2)
                                 if end_perp_length > 0:
                                     end_perp_lon /= end_perp_length
                                     end_perp_lat /= end_perp_length
                                 
                                 # 计算标记线的长度（使用固定的较小值）
                                 mark_length = 0.01  # 固定标记长度
                                 
                                 # 绘制起始点标记（竖着的横线）
                                 start_mark_start_lon = start_lon - start_perp_lon * mark_length
                                 start_mark_end_lon = start_lon + start_perp_lon * mark_length
                                 start_mark_start_lat = start_lat - start_perp_lat * mark_length
                                 start_mark_end_lat = start_lat + start_perp_lat * mark_length
                                 
                                 plt.plot([start_mark_start_lon, start_mark_end_lon], [start_mark_start_lat, start_mark_end_lat], 
                                          color='darkgray', linewidth=2, alpha=0.6, zorder=6)
                                 
                                 # 绘制结尾点标记（竖着的横线）
                                 end_mark_start_lon = end_lon - end_perp_lon * mark_length
                                 end_mark_end_lon = end_lon + end_perp_lon * mark_length
                                 end_mark_start_lat = end_lat - end_perp_lat * mark_length
                                 end_mark_end_lat = end_lat + end_perp_lat * mark_length
                                 
                                 plt.plot([end_mark_start_lon, end_mark_end_lon], [end_mark_start_lat, end_mark_end_lat], 
                                          color='darkgray', linewidth=2, alpha=0.6, zorder=6)
                            
                            print("使用原始点绘制断层线")
                    except Exception as fallback_error:
                        print(f"原始点绘制断层线也失败: {fallback_error}")
            else:
                print("没有断层数据或断层数据为空")
            
            # 如果仍然没有绘制任何等震线，尝试最后的备用方法
            if len(drawn_paths) == 0:
                print("=== 尝试最后的备用绘制方法 ===")
                # 获取所有唯一的烈度值，按烈度值排序
                all_intensities = self.line_df['intensity'].unique()
                print(f"所有烈度值: {all_intensities}")
                
                # 尝试按烈度值排序（从高到低）
                try:
                    # 将烈度值转换为数值进行排序
                    intensity_values = []
                    for intensity in all_intensities:
                        if intensity in intensity_str_to_value:
                            intensity_values.append((intensity, intensity_str_to_value[intensity]))
                        elif intensity in intensity_colors:
                            # 阿拉伯数字转数值
                            roman_to_num = {"X": 10, "IX": 9, "VIII": 8, "VII": 7, "VI": 6, "V": 5}
                            if intensity in roman_to_num:
                                intensity_values.append((intensity, roman_to_num[intensity]))
                            else:
                                intensity_values.append((intensity, 5))  # 默认值
                        else:
                            intensity_values.append((intensity, 5))  # 默认值
                    
                    # 按数值排序（从高到低）
                    intensity_values.sort(key=lambda x: x[1], reverse=True)
                    sorted_intensities = [item[0] for item in intensity_values]
                    print(f"排序后的烈度值: {sorted_intensities}")
                    
                    for intensity in sorted_intensities:
                        group = self.line_df[self.line_df["intensity"] == intensity]
                        print(f"备用方法 - 烈度 {intensity} 数据点数: {len(group)}")
                        if len(group) >= 1:  # 只要有数据点就尝试绘制
                            points = group[["Longtitude", "Latitude"]].values
                            try:
                                # 直接绘制原始点，不进行任何处理
                                color = 'black'
                                
                                plt.plot(points[:, 0], points[:, 1], color=color,
                                         label=f"Historical Intensity {intensity} (fallback)", linewidth=1, linestyle=':')
                                mid_idx = len(points) // 2
                                mid_x, mid_y = points[mid_idx, 0], points[mid_idx, 1]
                                plt.text(mid_x, mid_y, format_intensity_label_for_display(intensity),
                                         fontsize=14, color='black', ha='center', va='center',
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))
                                print(f"备用方法 - 成功绘制烈度 {intensity} 等震线")
                            except Exception as e:
                                print(f"备用方法 - 绘制烈度 {intensity} 等震线时出错: {e}")
                                continue
                except Exception as sort_error:
                    print(f"排序失败，使用原始顺序: {sort_error}")
                    # 如果排序失败，直接使用原始顺序
                    for intensity in all_intensities:
                        group = self.line_df[self.line_df["intensity"] == intensity]
                        if len(group) >= 1:
                            points = group[["Longtitude", "Latitude"]].values
                            try:
                                plt.plot(points[:, 0], points[:, 1], color='black',
                                         label=f"Historical Intensity {intensity} (fallback)", linewidth=1, linestyle=':')
                                mid_idx = len(points) // 2
                                mid_x, mid_y = points[mid_idx, 0], points[mid_idx, 1]
                                plt.text(mid_x, mid_y, format_intensity_label_for_display(intensity),
                                         fontsize=14, color='black', ha='center', va='center',
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))
                                print(f"备用方法 - 成功绘制烈度 {intensity} 等震线")
                            except Exception as e:
                                print(f"备用方法 - 绘制烈度 {intensity} 等震线时出错: {e}")
                                continue
            
            # 绘制城市点（更大字体、更小更浅的圆圈）
            print("绘制城市点...")
            # 城市点同样裁剪
            city_df = self.point_df
            if self.bbox is not None:
                city_df = city_df[(city_df["Longtitude"] >= all_lon_min) & (city_df["Longtitude"] <= all_lon_max) &
                                  (city_df["Latitude"] >= all_lat_min) & (city_df["Latitude"] <= all_lat_max)]
            plt.scatter(city_df["Longtitude"], city_df["Latitude"],
                        facecolors='lightgray', edgecolors='dimgray', marker='o',
                        s=18, linewidths=0.6, label="Cities")
            for _, row in city_df.iterrows():
                plt.text(row["Longtitude"] + 0.01, row["Latitude"], row["city_name"],
                         fontsize=17, ha='left', va='center')
            
            # 设置坐标轴
            plt.xlabel("Longitude", fontsize=17)
            plt.ylabel("Latitude", fontsize=17)
            plt.title(self.title_var.get(), fontsize=17)
            
            # 设置经纬度刻度间隔为0.5度，如果无法达到则只显示边框
            if self.bbox is not None:
                plt.xlim(all_lon_min, all_lon_max)
                plt.ylim(all_lat_min, all_lat_max)
            
            # 计算经纬度范围
            lon_range = all_lon_max - all_lon_min
            lat_range = all_lat_max - all_lat_min
            
            # 检查是否可以设置0.5度间隔
            if lon_range >= 0.5:
                # 计算0.5度间隔的刻度
                lon_start = np.floor(all_lon_min * 2) / 2.0
                lon_end = np.ceil(all_lon_max * 2) / 2.0
                lon_ticks = np.arange(lon_start, lon_end + 0.1, 0.5)
                lon_labels = [f"{x:.1f}°E" for x in lon_ticks]
            else:
                # 范围小于0.5度，只显示边框
                lon_ticks = []
                lon_labels = []
            
            if lat_range >= 0.5:
                # 计算0.5度间隔的刻度
                lat_start = np.floor(all_lat_min * 2) / 2.0
                lat_end = np.ceil(all_lat_max * 2) / 2.0
                lat_ticks = np.arange(lat_start, lat_end + 0.1, 0.5)
                lat_labels = [f"{y:.1f}°N" for y in lat_ticks]
            else:
                # 范围小于0.5度，只显示边框
                lat_ticks = []
                lat_labels = []
            
            plt.xticks(lon_ticks, lon_labels, rotation=45, fontsize=17)
            plt.yticks(lat_ticks, lat_labels, fontsize=17)
            
            # 根据下拉框选择设置图例位置
            legend_loc_map = {
                "左下角": "lower left",
                "右上角": "upper right",
            }
            legend_loc = legend_loc_map.get(self.legend_loc_var.get(), "lower left")
            plt.legend(loc=legend_loc, fontsize=17)
            # 移除内部网格线，只保留坐标轴
            plt.grid(False)
            
            # 添加颜色条
            cbar = plt.colorbar(contourf, shrink=0.8)
            cbar.set_label('Intensity (incl. X)', fontsize=17)
            
            # 设置颜色条刻度标签为罗马数字
            tick_labels = ['V', 'VI', 'VII', 'VIII', 'IX', 'X']
            tick_positions = [5, 6, 7, 8, 9, 10]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(17)
            
            # 调整布局并保存
            print("保存图像...")
            plt.tight_layout()
            # 统一保存目录
            output_dir = r"D:\PYTHON\exsim_manual_717\生成图像"
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as mk_err:
                print(f"创建输出目录失败，将尝试当前目录保存: {mk_err}")
                output_dir = os.getcwd()
            output_filename = f"{self.output_name_var.get()}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"图像已保存: {output_path}")
            messagebox.showinfo("完成", f"综合分析图已生成：{output_path}")

        except Exception as e:
            error_msg = f"创建图像时出错：{str(e)}\n\n详细错误信息：{type(e).__name__}"
            print(f"图像创建错误: {error_msg}")
            messagebox.showerror("错误", error_msg)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EarthquakeAnalysisGUI()
    app.run() 