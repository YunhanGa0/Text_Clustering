import re

import emoji
import pandas as pd


def clean_text(text):
    # 移除RT标记
    text = re.sub(r'^RT\s+@\w+:\s+', '', text)

    # 移除@用户名
    text = re.sub(r'@\w+', '', text)

    # 移除URL
    text = re.sub(r'https?://\S+', '', text)

    # 移除标签 (#开头的词)
    text = re.sub(r'#\w+', '', text)

    # 移除表情符号
    text = emoji.replace_emoji(text, '')

    # 移除特殊字符，只保留字母、数字、空格和基本标点
    text = re.sub(r'[^\w\s.,!?;:"\'-]', '', text)

    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()

    return text


# 读取原始数据
df = pd.read_csv('../corpus/拉非兹败选话题分析/twitter.csv')

# 检查是否有content列
if 'content' in df.columns:
    # 对content列应用清洗函数
    df['content'] = df['content'].astype(str).apply(clean_text)

    # 移除空行
    df = df[df['content'].str.strip().astype(bool)]

    # 去重
    # df = df.drop_duplicates(subset=['cleaned_content'])

    # 保存清洗后的数据
    df.to_csv('twitter.csv', index=False)
    print(f"清洗完成，共处理{len(df)}条数据")
else:
    print("找不到'content'列，请检查数据格式")