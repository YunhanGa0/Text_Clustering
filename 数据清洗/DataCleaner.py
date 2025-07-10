import re
import emoji
import pandas as pd
import hashlib
import json
from pathlib import Path


def clean_text(text):
    """
    清洗文本内容，移除URL、@用户名、标签、表情符号和特殊字符
    """
    # 确保文本是字符串类型
    text = str(text)

    # 移除URL
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # 移除@用户名
    text = re.sub(r'@\S+', '', text)

    # 移除标签
    text = re.sub(r'#\S+', '', text)

    # 移除表情符号
    text = emoji.replace_emoji(text, '')

    # 移除特殊字符，但保留中文和马来语等文字
    text = re.sub(r'[^\w\s.,!?;:"\'-]', '', text)

    # 规范化空格
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()

    return text


def generate_hash(text):
    """
    为文本生成MD5哈希值
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def clean_csv_content(input_file, output_file=None, content_column='content', remove_retweets=True):
    """
    清洗CSV文件中的content列

    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径，默认为在原文件名后加上"_cleaned"
    content_column (str): 内容列的名称，默认为'content'
    remove_retweets (bool): 是否删除转发，默认为True

    返回:
    pandas.DataFrame: 清洗后的数据框
    """
    # 设置默认输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")

    # 读取原始数据
    df = pd.read_csv(input_file)

    # 检查是否有指定的内容列
    if content_column in df.columns:
        # 过滤掉所有以RT开头的推文（如果需要）
        if remove_retweets:
            original_count = len(df)
            df = df[~df[content_column].astype(str).str.startswith('RT')]
            filtered_count = original_count - len(df)

        # 清洗内容
        df[content_column] = df[content_column].apply(clean_text)

        # 删除清洗后为空的行
        empty_count = df[~df[content_column].str.strip().astype(bool)].shape[0]
        df = df[df[content_column].str.strip().astype(bool)]

        # 添加哈希列
        df['content_hash'] = df[content_column].apply(generate_hash)

        # 按哈希值去重，保留第一条记录
        duplicate_count = df.duplicated(subset=['content_hash']).sum()
        df = df.drop_duplicates(subset=['content_hash'], keep='first')

        # 删除临时哈希列
        df = df.drop(columns=['content_hash'])

        # 保存清洗后的数据
        df.to_csv(output_file, index=False)

        # 输出处理信息
        print(f"清洗完成，已保存到: {output_file}")
        if remove_retweets:
            print(f"已过滤转发 {filtered_count} 条")
        print(f"已移除空内容 {empty_count} 条")
        print(f"已去除重复内容 {duplicate_count} 条")
        print(f"最终保留数据 {len(df)} 条")

        return df
    else:
        print(f"找不到'{content_column}'列，请检查数据格式")
        return None


def clean_jsonl_content(input_file, output_file=None, content_field='content', remove_retweets=True):
    """
    清洗JSONL文件中的content字段

    参数:
    input_file (str): 输入JSONL文件路径
    output_file (str): 输出JSONL文件路径，默认为在原文件名后加上"_cleaned"
    content_field (str): 内容字段的名称，默认为'content'
    remove_retweets (bool): 是否删除转发，默认为True

    返回:
    list: 清洗后的数据列表
    """
    # 设置默认输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")

    # 读取JSONL文件
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行")
    except Exception as e:
        print(f"读取文件出错: {str(e)}")
        return None

    original_count = len(data)
    cleaned_data = []
    empty_count = 0
    rt_count = 0
    type2_count = 0  # 统计article_type为2的条目数

    # 处理内容
    content_hashes = set()  # 用于去重

    for item in data:
        # 过滤article_type为2的条目
        if 'article_type' in item and item['article_type'] == 2:
            type2_count += 1
            continue

        # 检查是否有内容字段
        if content_field not in item:
            continue

        content = str(item[content_field])

        # 跳过转发（如果需要）
        if remove_retweets and content.startswith('RT '):
            rt_count += 1
            continue

        # 清洗内容
        cleaned_content = clean_text(content)

        # 跳过空内容
        if not cleaned_content.strip():
            empty_count += 1
            continue

        # 去重
        content_hash = generate_hash(cleaned_content)
        if content_hash in content_hashes:
            continue

        content_hashes.add(content_hash)

        # 只保留publish_time和content字段
        cleaned_item = {}
        if 'publish_time' in item:
            cleaned_item['publish_time'] = item['publish_time']
        cleaned_item[content_field] = cleaned_content

        # 添加到清洗后的数据集
        cleaned_data.append(cleaned_item)

    # 保存清洗后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"保存文件出错: {str(e)}")
        return cleaned_data

    # 输出处理信息
    duplicate_count = original_count - rt_count - empty_count - type2_count - len(cleaned_data)
    print(f"清洗完成，已保存到: {output_file}")
    print(f"已过滤article_type=2的条目 {type2_count} 条")
    if remove_retweets:
        print(f"已过滤转发 {rt_count} 条")
    print(f"已移除空内容 {empty_count} 条")
    print(f"已去除重复内容 {duplicate_count} 条")
    print(f"最终保留数据 {len(cleaned_data)} 条")

    return cleaned_data


# 使用示例
if __name__ == "__main__":
    # CSV
    # clean_csv_content('path/to/your/file.csv')

    # JSONL
    clean_jsonl_content('../corpus/附件/发文合集.jsonl', '发文合集_cleaned.jsonl')
    pass