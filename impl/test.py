import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# 加载文本数据
df = pd.read_csv("twitter.csv")
docs = df["content"].astype(str).tolist()
timestamps = pd.to_datetime(df["utime"])

# 使用多语言 BERT 模型作为嵌入器（支持中文、英文等）
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 构建 CountVectorizer（不设停用词，不做分词）
# 默认是空格分词，适合中英文混合场景（不使用中文分词）
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    max_features=10000
)

# ====== 4. 初始化 BERTopic 模型 ======
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    language="multilingual",  # 自动适应多语言环境
    calculate_probabilities=True,
    verbose=True
)

# 训练模型并生成主题标签
topics, probs = topic_model.fit_transform(docs)

# 打印 Top-N 主题标签
print("\n主题标签预览：")
print(topic_model.get_topic_info().head(10))

# 可视化：主题热度变化
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topic_model.visualize_topics_over_time(topics_over_time).show()

# 8. 可视化：Top-N 热门主题关键词条形图
topic_model.visualize_barchart(top_n_topics=10).show()

# 查看具体某主题的关键词词袋（可用于标签人工修正）
print("\n示例主题 Topic #1：")
for word, score in topic_model.get_topic(1):
    print(f"{word} ({score:.4f})")

# 保存模型，便于增量更新
topic_model.save("bertopic_multilingual_model")
