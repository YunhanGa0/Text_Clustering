import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


# 读取停用词
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

# 合并所有语言的停用词
stopwords = set()
stopwords.update(load_stopwords('../stopwords/Arabic.txt'))
stopwords.update(load_stopwords('../stopwords/Indonesian.txt'))
stopwords.update(load_stopwords('../stopwords/Malay.txt'))
stopwords.update(load_stopwords('../stopwords/English.txt'))
stopwords.update(load_stopwords('../stopwords/Chinese.txt'))
print(f"加载了 {len(stopwords)} 个停用词")

# ====== 1. 加载 JSONL 文件，字段应包含 'text' 和 'timestamp' ======
df = pd.read_json("../corpus/新闻数据/新闻数据.jsonl", lines=True)
# df = pd.read_csv("../corpus/拉非兹败选话题分析/oversea.csv")

# 只处理前500条数据
max_docs = 1000
df = df.head(max_docs)
print(f"仅处理前 {max_docs} 条数据")

docs = df["content"].astype(str).tolist()
timestamps = pd.to_datetime(df["publish_time"])
print(f"加载了 {len(docs)} 条评论数据，时间范围从 {timestamps.min()} 到 {timestamps.max()}")

# ====== 2. 嵌入模型：支持多语言的小模型 ======
#embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder="./bertopic_multilingual_model")
print("嵌入模型已加载：paraphrase-multilingual-MiniLM-L12-v2")

# ====== 3. 词频向量器：不设停用词，不设中文分词 ======
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    stop_words=list(stopwords)  # 添加停用词
)
print("词频向量器已配置：n-gram 范围 (1, 2)，最大特征数 10000")

# ====== 4. 初始化 BERTopic 模型 ======
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    language="multilingual",  # 自动适应多语言输入
    calculate_probabilities=True,
    verbose=True
)
print("BERTopic 模型已初始化，准备进行主题建模")

# ====== 5. 拟合主题模型 ======
topics, probs = topic_model.fit_transform(docs)
print("主题模型训练完成，生成了主题标签和概率分布")

# ====== 6. 输出前10个主题标签 ======
print("\n主题标签预览：")
print(topic_model.get_topic_info().head(10))

# ====== 7. 可视化热度变化（按时间切片）======
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topic_model.visualize_topics_over_time(topics_over_time).show()

# ====== 8. 可视化 Top-N 热门主题关键词条形图 ======
topic_model.visualize_barchart(
    top_n_topics=10,    # 显示前10个主题
    n_words=8,         # 每个主题显示8个关键词
).show()
topic_model.visualize_barchart(
    top_n_topics=10,    # 显示前10个主题
    n_words=8,         # 每个主题显示8个关键词
).write_html("barchart.html")

# ====== 9. 查看某个主题的关键词（如 Topic #1）======
print("\n示例主题 Topic #1：")
for word, score in topic_model.get_topic(1):
    print(f"{word} ({score:.4f})")

# ====== 10. 保存模型 ======
topic_model.save("bertopic_multilingual_model")
