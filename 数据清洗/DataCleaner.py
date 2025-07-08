import re
import emoji
import pandas as pd


def clean_text(text):
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


# 读取原始数据
df = pd.read_csv('../corpus/拉非兹败选话题分析/twitter.csv')

# 检查是否有content列
if 'content' in df.columns:
    # 过滤掉所有以RT开头的推文
    df = df[~df['content'].astype(str).str.startswith('RT')]

    # 清洗剩余推文内容
    df['content'] = df['content'].apply(clean_text)

    # 删除清洗后为空的行
    df = df[df['content'].str.strip().astype(bool)]

    # 保存清洗后的数据（直接覆盖原content列）
    df.to_csv('twitter_cleaned.csv', index=False)

    print(f"清洗完成，保留非转发推文共{len(df)}条")
else:
    print("找不到'content'列，请检查数据格式")