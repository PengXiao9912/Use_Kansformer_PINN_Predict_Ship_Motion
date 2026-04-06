import os
import re
import pandas as pd

def time_to_seconds(time_str):
    """将时间字符串转换为秒数（支持含日期格式）"""
    if " " in time_str:
        time_str = time_str.split()[-1]  # 移除日期部分
    parts = time_str.split(':')
    if len(parts) != 4:
        raise ValueError(f"无效时间格式: {time_str}")
    h, m, s, ms = map(int, parts)
    return h * 3600 + m * 60 + s + ms / 1000

def clean_numeric_str(x):
    """清洗含数字的字符串（支持负号和小数点）"""
    match = re.search(r'-?\d+\.?\d*', str(x))
    return float(match.group()) if match else None

def process_file(txt_path, excel_path, encoding="gbk"):
    """处理单个文件转换"""
    rows = []
    header = None
    base_values = {}  # 存储每个数据块的基准值
    first_header = True

    with open(txt_path, 'r', encoding=encoding, errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            
            # 表头检测逻辑
            if ("时间" in tokens[0] and 
                any("经度" in t for t in tokens) and 
                any("纬度" in t for t in tokens)):
                
                # 构造新表头（保留原始经纬度+新增相对列）
                if "-" in tokens[0]:  # 含日期的表头格式
                    new_header = [
                        "时间(秒)", 
                        "原始经度", "原始纬度",    # 原始列
                        "相对经度", "相对纬度",    # 新增相对列
                        tokens[4],               # 航向
                        "速度(m/s)", 
                        "艏摇角速度(°/s)"
                    ] + tokens[7:]  # 后续列保持原样
                else:               # 不含日期的表头格式
                    new_header = [
                        "时间(秒)",
                        "原始经度", "原始纬度",    # 原始列
                        "相对经度", "相对纬度",    # 新增相对列
                        tokens[3],               # 航向
                        "速度(m/s)", 
                        "艏摇角速度(°/s)"
                    ] + tokens[6:]  # 后续列保持原样
                
                header = new_header
                base_values = {}  # 重置基准值
                if first_header:
                    first_header = False
                continue

            # 数据行处理逻辑
            try:
                # 解析时间格式
                has_date = "-" in tokens[0]
                if has_date:
                    time_str = f"{tokens[0]} {tokens[1]}"
                    lon_idx, lat_idx = 2, 3
                    data_start = 4
                else:
                    time_str = tokens[0]
                    lon_idx, lat_idx = 1, 2
                    data_start = 3

                # 计算绝对时间
                abs_time = time_to_seconds(time_str)
                
                # 初始化基准值
                if not base_values:
                    base_values = {
                        "time": abs_time,
                        "lon": float(tokens[lon_idx]),
                        "lat": float(tokens[lat_idx])
                    }

                # 计算相对值
                rel_time = abs_time - base_values["time"]
                rel_lon = float(tokens[lon_idx]) - base_values["lon"]
                rel_lat = float(tokens[lat_idx]) - base_values["lat"]

                # 构建数据行
                row = [
                    rel_time,
                    float(tokens[lon_idx]),  # 原始经度
                    float(tokens[lat_idx]),  # 原始纬度
                    rel_lon,                # 相对经度
                    rel_lat,                # 相对纬度
                ]

                # 处理后续列数据
                if has_date:
                    speed_kn = clean_numeric_str(tokens[5])
                    yaw_rate = clean_numeric_str(tokens[6])
                    row += [
                        tokens[4],  # 航向
                        speed_kn * 0.5144 if speed_kn else None,
                        yaw_rate / 60.0 if yaw_rate else None
                    ] + tokens[7:]
                else:
                    speed_kn = clean_numeric_str(tokens[4])
                    yaw_rate = clean_numeric_str(tokens[5])
                    row += [
                        tokens[3],  # 航向
                        speed_kn * 0.5144 if speed_kn else None,
                        yaw_rate / 60.0 if yaw_rate else None
                    ] + tokens[6:]
                
                rows.append(row)
                
            except Exception as e:
                print(f"行 {line_num} 解析失败: {line}\n错误: {str(e)}")
                continue

    if not header:
        print(f"文件 {txt_path} 未检测到有效表头")
        return

    # 创建DataFrame并清洗数据
    df = pd.DataFrame(rows, columns=header)
    
    # 需要清洗的数值列（包含新增的6列）
    numeric_cols = [
        '航向', '速度(m/s)', '艏摇角速度(°/s)',
        '左舵角', '左转速', '左功率', '左转矩',
        '右舵角', '右转速', '右转功', '右转矩',
        '风速(m/s)', '风向(T/R)', '海底深(m)', '深(m)',
        'head', 'headRat', 'pitch', 'pitchRat', 'roll', 'rollRat'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_str)

    # 保存结果
    df.to_excel(excel_path, index=False)
    print(f"成功转换: {txt_path} -> {excel_path}")

def batch_convert_txt_to_excel(directory=".", encoding="gbk"):
    """批量转换目录下所有txt文件"""
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    if not txt_files:
        print(f"目录 {directory} 中没有找到txt文件")
        return
    
    for txt_file in txt_files:
        txt_path = os.path.join(directory, txt_file)
        excel_path = os.path.splitext(txt_path)[0] + ".xlsx"
        try:
            process_file(txt_path, excel_path, encoding)
        except Exception as e:
            print(f"文件 {txt_file} 转换失败: {str(e)}")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    file_encoding = sys.argv[2] if len(sys.argv) > 2 else "gbk"
    batch_convert_txt_to_excel(target_dir, file_encoding)