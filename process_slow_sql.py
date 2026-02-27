#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
慢SQL数据处理脚本
功能：
1. 读取CSV文件中的慢SQL统计数据
2. 按第5列（时间）排序
3. 根据第11列（SQL语句）去重，只保留首次出现的记录
4. 将结果保存为Excel文件
"""

import pandas as pd
from datetime import datetime
import sys


def process_slow_sql(input_csv_path, output_excel_path=None):
    """
    处理慢SQL数据
    
    Args:
        input_csv_path: 输入的CSV文件路径
        output_excel_path: 输出的Excel文件路径（可选，默认自动生成）
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_csv_path}")
        df = pd.read_csv(input_csv_path, encoding='utf-8')
        
        print(f"原始数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        
        # 检查列数是否足够
        if len(df.columns) < 11:
            print(f"警告: CSV文件只有 {len(df.columns)} 列，需要至少11列")
            return
        
        # 获取列名（索引从0开始，第5列是索引4，第11列是索引10）
        time_column = df.columns[4]  # 第5列（时间）
        sql_column = df.columns[10]  # 第11列（SQL）
        
        print(f"时间列: {time_column}")
        print(f"SQL列: {sql_column}")
        
        # 按时间列倒序排序
        print("正在按时间倒序排序...")
        df_sorted = df.sort_values(by=time_column, ascending=False)
        
        # 根据SQL列去重，保留第一次出现的记录
        print("正在根据SQL去重...")
        df_unique = df_sorted.drop_duplicates(subset=[sql_column], keep='first')
        
        print(f"去重后数据行数: {len(df_unique)}")
        print(f"去除了 {len(df_sorted) - len(df_unique)} 条重复记录")
        
        # 生成输出文件名
        if output_excel_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_excel_path = f"slow_sql_processed_{timestamp}.xlsx"
        
        # 保存为Excel文件
        print(f"正在保存到Excel文件: {output_excel_path}")
        df_unique.to_excel(output_excel_path, index=False, engine='openpyxl')
        
        print(f"✓ 处理完成！结果已保存到: {output_excel_path}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_csv_path}")
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 使用示例
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <输入CSV文件路径> [输出Excel文件路径]")
        print("\n示例:")
        print(f"  python {sys.argv[0]} slow_sql_data.csv")
        print(f"  python {sys.argv[0]} slow_sql_data.csv output.xlsx")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        process_slow_sql(input_file, output_file)
