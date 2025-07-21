import json


def merge_jsonl_files(file1_path, file2_path, output_path):
    """
    合并两个JSONL文件到一个新的JSONL文件

    参数:
        file1_path (str): 第一个JSONL文件路径
        file2_path (str): 第二个JSONL文件路径
        output_path (str): 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # 处理第一个文件
        with open(file1_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    # 验证JSON格式并保留原始行
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"警告: 文件 {file1_path} 中跳过无效的JSON行: {line.strip()}")

        # 处理第二个文件
        with open(file2_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    # 验证JSON格式并保留原始行
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"警告: 文件 {file2_path} 中跳过无效的JSON行: {line.strip()}")

    print(f"成功合并文件到 {output_path}")


# 使用示例
if __name__ == "__main__":
    file1 = "../corpus/附件/发文合集.jsonl"  # 第一个输入文件路径
    file2 = "../orpus/新闻数据/新闻数据.jsonl"  # 第二个输入文件路径
    output = "merged_output.jsonl"  # 输出文件路径

    merge_jsonl_files(file1, file2, output)