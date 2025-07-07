import csv
import os


def extract_column(input_file, output_file, column_name=None, column_index=None):
    """
    从CSV文件中提取指定列并写入新的CSV文件

    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        column_name: 要提取的列名称（与column_index二选一）
        column_index: 要提取的列索引（与column_name二选一）
    """
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在")
        return False

    if column_name is None and column_index is None:
        print("错误: 必须指定列名称或列索引")
        return False

    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)  # 读取表头

            # 如果提供了列名，找到对应的索引
            if column_name is not None:
                if column_name not in headers:
                    print(f"错误: 列名 '{column_name}' 不存在于输入文件中")
                    return False
                column_index = headers.index(column_name)

            # 提取数据
            extracted_data = []
            extracted_data.append([headers[column_index]])  # 添加表头

            for row in reader:
                if column_index < len(row):
                    extracted_data.append([row[column_index]])

        # 写入新文件
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(extracted_data)

        print(f"成功: 已将列 '{headers[column_index]}' 提取到文件 '{output_file}'")
        return True

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


if __name__ == "__main__":
    # 直接使用文件地址和列名
    input_file = "../corpus/拉非兹败选话题分析/twitter.csv"  # 替换为实际的CSV文件路径
    output_file = "提取结果.csv"  # 输出文件名
    column_name = "content"  # 替换为你要提取的列名

    # 执行提取操作
    extract_column(input_file, output_file, column_name=column_name)