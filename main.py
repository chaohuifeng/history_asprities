"""
EXSIM 参数生成器 - 替换主要信息并生成不同数组，支持并行处理
EXSIM Parameter Generator - Replace main information and generate different arrays, supports parallel processing
"""

import os
import shutil
import math
from tkinter import filedialog, messagebox, Tk, Label, Entry, Button, StringVar, Frame

# 需要提取并替换的行号列表 / List of line numbers to extract and replace
target_line_numbers = [30, 32, 41, 56, 58, 87, 91, 95, 112, 124, 194]

# 初始化全局参数字典 / Initialize global parameter dictionary
global_params = {}
param_entries = {}


def browse_file():
    """
    浏览并选择参数文件 / Browse and select parameter file
    """
    file_path = filedialog.askopenfilename(filetypes=[("Params File", "*.params")])
    if file_path:
        param_file_path.set(file_path)
        load_params(file_path)


def load_params(file_path):
    """
    加载参数文件，并提取需要修改的行 / Load parameter file and extract lines to modify
    
    参数 / Parameters:
        file_path: 参数文件路径 / Parameter file path
    """
    try:
        if not os.path.exists(file_path):
            raise ValueError("文件路径无效！ / Invalid file path!")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # 提取目标行内容 / Extract target line content
        global_params.clear()

        for idx, line in enumerate(lines):
            if idx + 1 in target_line_numbers:  # 检查是否在目标行号列表中 / Check if in target line number list
                line = line.strip()  # 去掉多余空格 / Remove extra spaces
                global_params[idx + 1] = line  # 使用行号作为键存储完整行内容 / Use line number as key to store full line content

        # 在界面中显示可编辑内容 / Display editable content in interface
        for widget in param_frame.winfo_children():
            widget.destroy()

        for idx, (line_num, content) in enumerate(global_params.items()):
            frame = Frame(param_frame)
            frame.grid(row=idx, column=0, sticky="w")

            Label(frame, text=f"行{line_num}:").grid(row=0, column=0, sticky="e")
            entry = Entry(frame, width=50)
            entry.insert(0, content)
            entry.grid(row=0, column=1)
            param_entries[line_num] = entry

        messagebox.showinfo("成功", f"参数文件 {os.path.basename(file_path)} 已加载！ / Parameter file {os.path.basename(file_path)} loaded successfully!")

    except Exception as e:
        messagebox.showerror("错误", f"加载参数文件时出错: {e} / Error loading parameter file: {e}")


def save_and_generate():
    """
    统一流程：生成经纬度站点，替换指定行内容，生成多个文件
    Unified process: Generate latitude/longitude sites, replace specified line content, generate multiple files
    """
    try:
        # 获取用户输入或使用默认值 / Get user input or use default values
        site_min_lat = float(entry_min_lat.get() or 22.8)
        site_max_lat = float(entry_max_lat.get() or 27.3)
        site_min_lon = float(entry_min_lon.get() or 102)
        site_max_lon = float(entry_max_lon.get() or 104.3)
        step_lat = float(entry_step_lat.get() or 0.1)
        step_lon = float(entry_step_lon.get() or 0.1)
        arrays = int(entry_arrays.get() or 10)

        # 生成所有站点坐标 / Generate all site coordinates
        latitudes = [round(site_min_lat + i * step_lat, 4)
                     for i in range(int((site_max_lat - site_min_lat) / step_lat) + 1)]
        longitudes = [round(site_min_lon + i * step_lon, 4)
                      for i in range(int((site_max_lon - site_min_lon) / step_lon) + 1)]
        total_sites = len(latitudes) * len(longitudes)
        sites = [(lat, lon) for lat in latitudes for lon in longitudes]

        print(f"total sites: {total_sites}")

        # 分组站点 / Group sites
        sites_per_array = math.ceil(total_sites / arrays)
        site_arrays = [sites[i:i + sites_per_array] for i in range(0, total_sites, sites_per_array)]

        print(f"Sites per array: {sites_per_array}")

        # 获取输出文件夹路径 / Get output folder path
        output_folder_path = entry_output_path.get().strip()
        if not output_folder_path:
            raise ValueError("输出路径不能为空！ / Output path cannot be empty!")
        os.makedirs(output_folder_path, exist_ok=True)

        # 生成多个参数文件 / Generate multiple parameter files
        for idx, site_array in enumerate(site_arrays):
            subfolder = os.path.join(output_folder_path, f"Array_{idx + 1}")
            os.makedirs(subfolder, exist_ok=True)

            # 复制原始参数文件 / Copy original parameter file
            params_path = os.path.join(subfolder, "exsim_dmb.params")
            shutil.copy(param_file_path.get(), params_path)

            # 修改参数文件基础部分 / Modify parameter file base section
            modify_params(params_path, site_array)

            # 复制必要的参数文件 / Copy necessary parameter files
            for file_name in ["crustal_amps_sample.txt", "site_amps_sample.txt", "exsim_dmb.exe", "slip_weights.txt"]:
                src = os.path.join(os.path.dirname(param_file_path.get()), file_name)
                dest = os.path.join(subfolder, file_name)
                if os.path.exists(src):
                    shutil.copy(src, dest)
                else:
                    messagebox.showwarning("警告", f"文件 {file_name} 未找到，无法复制。 / File {file_name} not found, cannot copy.")

        messagebox.showinfo("成功", f"所有参数文件已生成并保存至：{output_folder_path} / All parameter files generated and saved to: {output_folder_path}")

    except Exception as e:
        messagebox.showerror("错误", f"生成文件时出错: {e} / Error generating files: {e}")


def modify_params(file_path, site_array):
    """
    修改参数文件并插入站点经纬度 / Modify parameter file and insert site latitude/longitude
    
    参数 / Parameters:
        file_path: 参数文件路径 / Parameter file path
        site_array: 站点坐标数组 / Site coordinate array
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # 修改指定行内容 / Modify specified line content
        for line_num, entry in param_entries.items():
            if line_num - 1 < len(lines):
                lines[line_num - 1] = entry.get() + "\n"

        # 保留前230行内容不变 / Keep first 230 lines unchanged
        if len(lines) > 230:
            lines = lines[:230]

        # 插入站点经纬度，从231行开始 / Insert site latitude/longitude, starting from line 231
        for lat, lon in site_array:
            lines.append(f"{lat:.2f}       {lon:.2f}\n")

        # 添加 stop 行 / Add stop line
        lines.append("stop\n")

        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)

    except Exception as e:
        messagebox.showerror("错误", f"修改参数文件时出错: {e} / Error modifying parameter file: {e}")


# 创建主界面 / Create main interface
root = Tk()
root.title("EXSIM 参数生成器 / EXSIM Parameter Generator")

# 参数输入框 / Parameter input box
Label(root, text="参数文件路径:").grid(row=0, column=0, sticky="e")
param_file_path = StringVar()
entry_file = Entry(root, textvariable=param_file_path, width=50)
entry_file.grid(row=0, column=1)
btn_browse = Button(root, text="浏览", command=browse_file)
btn_browse.grid(row=0, column=2)

Label(root, text="基础参数 (可修改):").grid(row=1, column=0, columnspan=3, sticky="w")
param_frame = Frame(root)
param_frame.grid(row=2, column=0, columnspan=3)

# 经纬度和步长输入 / Latitude, longitude and step input
Label(root, text="站点最小纬度:").grid(row=3, column=0, sticky="e")
entry_min_lat = Entry(root)
entry_min_lat.grid(row=3, column=1)
entry_min_lat.insert(0, str(22.8))  # 转换为字符串 / Convert to string

Label(root, text="站点最大纬度:").grid(row=4, column=0, sticky="e")
entry_max_lat = Entry(root)
entry_max_lat.grid(row=4, column=1)
entry_max_lat.insert(0, str(27.3))  # 转换为字符串 / Convert to string

Label(root, text="站点最小经度:").grid(row=5, column=0, sticky="e")
entry_min_lon = Entry(root)
entry_min_lon.grid(row=5, column=1)
entry_min_lon.insert(0, str(101.7))  # 转换为字符串 / Convert to string

Label(root, text="站点最大经度:").grid(row=6, column=0, sticky="e")
entry_max_lon = Entry(root)
entry_max_lon.grid(row=6, column=1)
entry_max_lon.insert(0, str(104.3))  # 转换为字符串 / Convert to string

Label(root, text="纬度步长:").grid(row=7, column=0, sticky="e")
entry_step_lat = Entry(root)
entry_step_lat.grid(row=7, column=1)
entry_step_lat.insert(0, str(0.1))  # 转换为字符串 / Convert to string

Label(root, text="经度步长:").grid(row=8, column=0, sticky="e")
entry_step_lon = Entry(root)
entry_step_lon.grid(row=8, column=1)
entry_step_lon.insert(0, str(0.1))  # 转换为字符串 / Convert to string

Label(root, text="分组数量:").grid(row=9, column=0, sticky="e")
entry_arrays = Entry(root)
entry_arrays.grid(row=9, column=1)
entry_arrays.insert(0, str(10))  # 转换为字符串 / Convert to string

Label(root, text="输出路径:").grid(row=10, column=0, sticky="e")
entry_output_path = Entry(root, width=50)
entry_output_path.grid(row=10, column=1)

btn_save = Button(root, text="保存并生成", command=save_and_generate)
btn_save.grid(row=11, column=0, columnspan=3)

root.mainloop()
