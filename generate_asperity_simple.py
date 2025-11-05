#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
地震凹凸体模型生成器 - 简化版本
Earthquake Asperity Model Generator - Simplified Version
Chaohui.Feng 2025.10.01
mail:fengchaohui23@mails.ucas.ac.cn

功能：
- 生成指定大小的凹凸体（断层1/3大小）
- 凹凸体滑移量占比0.5，背景滑移量占比0.25
- 保持能量平衡，确保总地震矩正确
- 支持凹凸体位置控制（左、中、右）
- 输出滑移分布文件和可视化图片

使用方法：
python generate_asperity_simple.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 参数设置区域 - 在这里修改参数 / Parameter Setting Area - Modify parameters here
# =============================================================================

# 地震参数 / Earthquake parameters
Mw = 7.0          # 矩震级 / Moment magnitude
L = 64.0          # 断层长度 (km) / Fault length (km)
W = 20.0           # 断层宽度 (km) / Fault width (km)
dx = 2.0           # 子断层长度 (km) / Subfault length (km)
dz = 2.0           # 子断层宽度 (km) / Subfault width (km)

# 凹凸体位置选择 ('left', 'center', 'right') / Asperity position selection ('left', 'center', 'right')
# 可以修改为单个位置，如：positions = ['center'] / Can modify to single position, e.g.: positions = ['center']
# 或者多个位置，如：positions = ['left', 'center', 'right'] / Or multiple positions, e.g.: positions = ['left', 'center', 'right']
positions = ['left', 'center', 'right']

# 是否生成可视化图片 / Whether to generate visualization plots
generate_plots = True

# 是否保存滑移权重文件 / Whether to save slip weight files
save_slip_files = True

# =============================================================================

def calculate_seismic_moment(Mw):
    """
    根据矩震级计算地震矩 / Calculate seismic moment from moment magnitude
    使用Hanks和Kanamori（1979）提出的关系式 / Using the relationship proposed by Hanks and Kanamori (1979)
    
    参数 / Parameters:
    Mw: 矩震级 / Moment magnitude
    
    返回 / Returns:
    Mo: 地震矩 (N⋅m) / Seismic moment (N⋅m)
    """
    # Hanks和Kanamori（1979）公式: log10(M0×10^7) = 1.5Mw + 16.1
    # Hanks and Kanamori (1979) formula: log10(M0×10^7) = 1.5Mw + 16.1
    # 因此: log10(M0) = 1.5Mw + 16.1 - 7 = 1.5Mw + 9.1
    # Therefore: log10(M0) = 1.5Mw + 16.1 - 7 = 1.5Mw + 9.1
    log_Mo = 1.5 * Mw + 9.1
    Mo = 10 ** log_Mo  # N⋅m
    return Mo

def generate_asperity_model(Mw, L, W, dx, dz, asperity_position='center'):
    """
    生成凹凸体模型 / Generate asperity model
    
    参数 / Parameters:
    Mw: 矩震级 / Moment magnitude
    L: 断层长度 (km) / Fault length (km)
    W: 断层宽度 (km) / Fault width (km)
    dx: 子断层长度 (km) / Subfault length (km)
    dz: 子断层宽度 (km) / Subfault width (km)
    asperity_position: 凹凸体位置 ('left', 'center', 'right') / Asperity position ('left', 'center', 'right')
    
    返回 / Returns:
    slip: 滑移分布数组 (nz, nx) / Slip distribution array (nz, nx)
    filename_prefix: 文件名前缀 / Filename prefix
    """
    
    print("=" * 80)
    print("地震凹凸体模型生成器")
    print("Earthquake Asperity Model Generator")
    print("=" * 80)
    
    # 计算地震矩 / Calculate seismic moment
    Mo = calculate_seismic_moment(Mw)
    
    # 计算网格尺寸 / Calculate grid dimensions
    nx = int(L / dx)  # 沿走向网格点数 / Number of grid points along strike
    nz = int(W / dz)  # 沿倾向网格点数 / Number of grid points along dip
    
    print(f"地震参数:")
    print(f"  矩震级: {Mw}")
    print(f"  断层长度: {L} km")
    print(f"  断层宽度: {W} km")
    print(f"  子断层尺寸: {dx}*{dz} km")
    print(f"  网格尺寸: {nz}行 * {nx}列")
    print(f"  总地震矩: {Mo:.2e} N*m")
    
    # 计算断层总面积 / Calculate total fault area
    total_area = L * W  # km²
    
    # 凹凸体面积（断层长度的1/3，深度5-15km） / Asperity area (1/3 of fault length, depth 5-15km)
    asperity_length = L / 3.0  # km - 断层长度的1/3 / km - 1/3 of fault length
    asperity_depth_start = 5.0  # km - 凹凸体起始深度 / km - Asperity starting depth
    asperity_depth_end = 15.0   # km - 凹凸体结束深度 / km - Asperity ending depth
    asperity_width = asperity_depth_end - asperity_depth_start  # km - 凹凸体深度范围 / km - Asperity depth range
    asperity_area = asperity_length * asperity_width  # km²
    background_area = total_area - asperity_area  # km²
    
    print(f"面积分布:")
    print(f"  断层总面积: {total_area:.1f} km2")
    print(f"  凹凸体面积: {asperity_area:.1f} km2 ({asperity_area/total_area*100:.1f}%)")
    print(f"  背景区域面积: {background_area:.1f} km2 ({background_area/total_area*100:.1f}%)")
    
    print(f"凹凸体尺寸: {asperity_length:.1f}*{asperity_width:.1f} km (长度1/3，深度{asperity_depth_start}-{asperity_depth_end}km)")
    
    # 根据位置确定凹凸体范围（深度5-15km） / Determine asperity range based on position (depth 5-15km)
    if asperity_position == 'left':
        x_start = 0.0  # 断层起始位置 / Fault starting position
        x_end = asperity_length  # 断层长度的1/3处 / At 1/3 of fault length
    elif asperity_position == 'center':
        x_start = L / 3.0  # 断层长度的1/3处 / At 1/3 of fault length
        x_end = 2 * L / 3.0  # 断层长度的2/3处 / At 2/3 of fault length
    elif asperity_position == 'right':
        x_start = 2 * L / 3.0  # 断层长度的2/3处 / At 2/3 of fault length
        x_end = L  # 断层结束位置 / Fault ending position
    else:
        raise ValueError("asperity_position must be 'left', 'center', or 'right'")
    
    # 凹凸体深度范围5-15km / Asperity depth range 5-15km
    z_start = asperity_depth_start  # 5km深度 / 5km depth
    z_end = asperity_depth_end  # 15km深度 / 15km depth
    
    print(f"凹凸体位置: {asperity_position} - 范围: x=[{x_start:.1f}, {x_end:.1f}] km, z=[{z_start:.1f}, {z_end:.1f}] km")
    
    # 创建坐标网格 / Create coordinate grid
    x = np.linspace(0, L, nx)  # 沿走向坐标 (km) / Coordinates along strike (km)
    z = np.linspace(0, W, nz)  # 沿倾向坐标 (km) / Coordinates along dip (km)
    X, Z = np.meshgrid(x, z)
    
    # 初始化滑移分布 / Initialize slip distribution
    slip = np.zeros((nz, nx))
    
    # 计算滑移量以满足能量平衡 / Calculate slip to satisfy energy balance
    # 使用剪切模量 μ = 3.3×10^10 Pa / Using shear modulus μ = 3.3×10^10 Pa
    mu = 3.3e10  # Pa
    
    # 计算初始滑移量D = M/(μ⋅W⋅L) / Calculate initial slip D = M/(μ⋅W⋅L)
    # 将断层尺寸从km转换为m / Convert fault dimensions from km to m
    L_m = L * 1000  # m
    W_m = W * 1000  # m
    
    # 计算初始滑移量 / Calculate initial slip
    D_initial = Mo / (mu * W_m * L_m)
    
    # 凹凸体滑移量占比0.5，背景滑移量占比0.25 / Asperity slip ratio 0.5, background slip ratio 0.25
    # 设背景滑移量为 D_bg，则凹凸体滑移量为 2*D_bg / Let background slip be D_bg, then asperity slip is 2*D_bg
    # 总地震矩 = μ * (D_bg * A_bg + 2*D_bg * A_as) / Total seismic moment = μ * (D_bg * A_bg + 2*D_bg * A_as)
    # Mo = μ * D_bg * (A_bg + 2*A_as) / Mo = μ * D_bg * (A_bg + 2*A_as)
    
    # 将面积从km²转换为m² / Convert area from km² to m²
    asperity_area_m2 = asperity_area * 1e6  # m²
    background_area_m2 = background_area * 1e6  # m²
    
    # 计算背景滑移量 / Calculate background slip
    # 使用更精确的能量平衡计算 / Using more precise energy balance calculation
    # Mo = μ * (D_bg * A_bg + D_as * A_as) / Mo = μ * (D_bg * A_bg + D_as * A_as)
    # 设 D_as = 2 * D_bg，则 Mo = μ * D_bg * (A_bg + 2 * A_as) / Let D_as = 2 * D_bg, then Mo = μ * D_bg * (A_bg + 2 * A_as)
    D_bg = Mo / (mu * (background_area_m2 + 2 * asperity_area_m2))
    D_as = 2 * D_bg  # 凹凸体滑移量 / Asperity slip
    
    # 微调滑移量以更好地保持能量守恒 / Fine-tune slip to better maintain energy conservation
    total_moment_check = mu * (D_bg * background_area_m2 + D_as * asperity_area_m2)
    if abs(total_moment_check - Mo) / Mo > 0.01:  # 如果误差>1% / If error > 1%
        # 调整滑移量比例 / Adjust slip ratio
        scale_factor = Mo / total_moment_check
        D_bg *= scale_factor
        D_as *= scale_factor
    
    print(f"滑移量计算:")
    print(f"  初始滑移量 (D = M/(mu*W*L)): {D_initial:.3f} m")
    print(f"  背景滑移量: {D_bg:.3f} m")
    print(f"  凹凸体滑移量: {D_as:.3f} m")
    print(f"  滑移量比例: 背景={D_bg/D_as:.2f}, 凹凸体=1.0")
    
    # 设置背景滑移量 / Set background slip
    slip[:, :] = D_bg
    
    # 设置凹凸体区域（深度5-15km） / Set asperity region (depth 5-15km)
    asperity_points = 0
    for i in range(nz):
        for j in range(nx):
            x_pos = x[j]
            z_pos = z[i]
            
            # 检查是否在凹凸体范围内（深度5-15km） / Check if within asperity range (depth 5-15km)
            if (x_start <= x_pos <= x_end and z_start <= z_pos <= z_end):
                slip[i, j] = D_as
                asperity_points += 1
    
    # 计算实际凹凸体面积 / Calculate actual asperity area
    actual_asperity_area = asperity_points * dx * dz
    actual_background_area = total_area - actual_asperity_area
    
    print(f"实际面积分布:")
    print(f"  实际凹凸体面积: {actual_asperity_area:.1f} km2 ({actual_asperity_area/total_area*100:.1f}%)")
    print(f"  实际背景面积: {actual_background_area:.1f} km2 ({actual_background_area/total_area*100:.1f}%)")
    
    # 使用实际面积重新计算滑移量以保持能量守恒 / Recalculate slip using actual area to maintain energy conservation
    actual_asperity_area_m2 = actual_asperity_area * 1e6  # m²
    actual_background_area_m2 = actual_background_area * 1e6  # m²
    
    # 重新计算滑移量 / Recalculate slip
    D_bg_corrected = Mo / (mu * (actual_background_area_m2 + 2 * actual_asperity_area_m2))
    D_as_corrected = 2 * D_bg_corrected
    
    # 更新滑移分布 / Update slip distribution
    slip[:, :] = D_bg_corrected
    for i in range(nz):
        for j in range(nx):
            x_pos = x[j]
            z_pos = z[i]
            if (x_start <= x_pos <= x_end and z_start <= z_pos <= z_end):
                slip[i, j] = D_as_corrected
    
    # 验证能量平衡 / Verify energy balance
    total_moment = mu * (np.sum(slip) * dx * dz * 1e6)  # N*m
    moment_error = abs(total_moment - Mo) / Mo * 100
    
    print(f"能量平衡验证:")
    print(f"  目标地震矩: {Mo:.2e} N*m")
    print(f"  计算地震矩: {total_moment:.2e} N*m")
    print(f"  误差: {moment_error:.2f}%")
    
    if moment_error > 1.0:
        print(f"警告: 能量平衡误差较大 ({moment_error:.2f}%)")
    else:
        print(f"能量平衡验证通过")
    
    # 更新返回的滑移量值 / Update returned slip values
    D_bg = D_bg_corrected
    D_as = D_as_corrected
    
    # 生成文件名前缀 / Generate filename prefix
    filename_prefix = f"Mw{Mw}_L{L}km_W{W}km_{asperity_position}"
    
    return slip, X, Z, filename_prefix, D_bg, D_as

def save_slip_weights(slip, filename_prefix):
    """
    保存滑移权重文件 / Save slip weight file
    
    参数 / Parameters:
        slip: 滑移分布数组 / Slip distribution array
        filename_prefix: 文件名前缀 / Filename prefix
    
    返回 / Returns:
        filename: 保存的文件名 / Saved filename
    """
    filename = f'{filename_prefix}_slip_weights.txt'
    np.savetxt(filename, slip, fmt='%.6f')
    print(f"滑移权重文件已保存: {filename} / Slip weight file saved: {filename}")
    return filename

def create_visualization(slip, X, Z, filename_prefix, asperity_position, D_bg, D_as):
    """
    创建可视化图片 / Create visualization plot
    
    参数 / Parameters:
        slip: 滑移分布数组 / Slip distribution array
        X: 经度网格 / Longitude grid
        Z: 纬度网格 / Latitude grid
        filename_prefix: 文件名前缀 / Filename prefix
        asperity_position: 凹凸体位置 / Asperity position
        D_bg: 背景滑移量 / Background slip
        D_as: 凹凸体滑移量 / Asperity slip
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 设置标题
        position_names = {'left': '左侧', 'center': '中间', 'right': '右侧'}
        title = f'凹凸体位置: {position_names[asperity_position]} - 滑移分布'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 使用连续色阶
        cmap = plt.colormaps['RdYlBu_r']  # 红色-黄色-蓝色，反向
        
        # 绘制滑移分布
        im = ax.contourf(X, Z, slip, levels=20, cmap=cmap)
        
        # 设置坐标轴
        ax.set_xlabel('Rupture length (km)', fontsize=12)
        ax.set_ylabel('Rupture width (km)', fontsize=12)
        
        # 添加等值线
        contours = ax.contour(X, Z, slip, levels=10, colors='black', linewidths=0.8, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # 添加色带
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Slip (m)', rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # 添加统计信息
        max_slip = np.max(slip)
        mean_slip = np.mean(slip)
        
        info_text = f'最大滑移: {max_slip:.3f} m\n平均滑移: {mean_slip:.3f} m\n滑移比例: {D_as/D_bg:.1f}:1'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # 保存图形
        plot_filename = f'{filename_prefix}_plots.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"可视化图片已保存: {plot_filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"生成可视化图片时出错: {str(e)}")

def main():
    """
    主函数 - 生成指定位置的凹凸体模型 / Main function - Generate asperity model at specified positions
    """
    
    print("生成凹凸体模型...")
    print(f"参数: Mw={Mw}, L={L}km, W={W}km, 子断层={dx}*{dz}km")
    print(f"位置: {positions}")
    print(f"生成图片: {generate_plots}, 保存文件: {save_slip_files}")
    
    for pos in positions:
        print(f"\n{'='*60}")
        print(f"生成{pos}位置凹凸体模型...")
        print(f"{'='*60}")
        
        # 生成模型
        slip, X, Z, filename_prefix, D_bg, D_as = generate_asperity_model(
            Mw, L, W, dx, dz, pos)
        
        # 保存文件
        if save_slip_files:
            save_slip_weights(slip, filename_prefix)
        
        # 创建可视化
        if generate_plots:
            create_visualization(slip, X, Z, filename_prefix, pos, D_bg, D_as)
        
        print(f"{pos}位置模型生成完成")
    
    print(f"\n{'='*80}")
    print("凹凸体模型生成完成！")
    print(f"{'='*80}")
    print("生成的文件:")
    for pos in positions:
        prefix = f"Mw{Mw}_L{L}km_W{W}km_{pos}"
        if save_slip_files:
            print(f"  - {prefix}_slip_weights.txt")
        if generate_plots:
            print(f"  - {prefix}_plots.png")

if __name__ == "__main__":
    main()
