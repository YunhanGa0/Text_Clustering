import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
#df = pd.read_json("../corpus/新闻数据/新闻数据.jsonl", lines=True)
df = pd.read_csv("../数据清洗/twitter.csv")

# 只处理前500条数据
"""max_docs = 500
df = df.head(max_docs)
print(f"仅处理前 {max_docs} 条数据")"""

docs = df["content"].astype(str).tolist()
timestamps = pd.to_datetime(df["utime"])
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

# ====== 11. 生成聚类可视化 ======
"""print("\n生成聚类可视化...")

# 获取文档嵌入
embeddings = topic_model._extract_embeddings(docs)
print(f"嵌入向量形状: {embeddings.shape}")

# 使用 UMAP 降维到二维空间用于可视化
umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_embeddings = umap_model.fit_transform(embeddings)
print("UMAP 降维完成")

# 可视化聚类结果
fig = make_subplots(rows=1, cols=1)

# 过滤掉噪声主题 (-1)
valid_indices = [i for i, topic in enumerate(topics) if topic != -1]
valid_topics = [topics[i] for i in valid_indices]
valid_embeddings = umap_embeddings[valid_indices]

# 获取有效的主题列表和颜色映射
unique_topics = sorted(set(valid_topics))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_topics)))
topic_colors = {t: f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for t, c in zip(unique_topics, colors)}

# 为每个主题添加一个散点图层
for topic in unique_topics:
    indices = [i for i, t in enumerate(valid_topics) if t == topic]
    cluster_points = valid_embeddings[indices]

    # 获取该主题的代表性关键词
    if topic in topic_model.get_topics():
        keywords = [word for word, _ in topic_model.get_topic(topic)[:3]]
        topic_name = f"主题 {topic}: {', '.join(keywords)}"
    else:
        topic_name = f"主题 {topic}"

    fig.add_trace(
        go.Scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            mode='markers',
            marker=dict(color=topic_colors[topic], size=8),
            name=topic_name,
            hovertemplate="<b>" + topic_name + "</b><br>%{text}",
            text=[docs[valid_indices[i]][:50] + "..." for i in indices]  # 显示文本前50个字符
        )
    )

# 添加主题中心点
topic_info = topic_model.get_topic_info()
topic_info = topic_info[topic_info["Topic"] != -1]  # 过滤掉噪声主题

# 计算每个主题的中心点
topic_centers = {}
for topic in unique_topics:
    indices = [i for i, t in enumerate(valid_topics) if t == topic]
    if indices:
        center = np.mean(valid_embeddings[indices], axis=0)
        topic_centers[topic] = center

# 添加中心点
center_x = [topic_centers[t][0] for t in topic_centers]
center_y = [topic_centers[t][1] for t in topic_centers]
center_text = [f"主题 {t} 中心" for t in topic_centers]

fig.add_trace(
    go.Scatter(
        x=center_x,
        y=center_y,
        mode='markers+text',
        marker=dict(color='black', symbol='star', size=15, line=dict(width=2, color='white')),
        text=[f"主题 {t}" for t in topic_centers.keys()],
        name="主题中心",
        textposition="top center",
        hoverinfo="text",
        hovertext=center_text
    )
)

# 设置布局
fig.update_layout(
    title="文本聚类可视化（UMAP降维）",
    xaxis_title="UMAP 维度 1",
    yaxis_title="UMAP 维度 2",
    legend_title="主题分类",
    width=1200,
    height=800,
    hovermode="closest"
)

# 显示并保存图表
fig.show()
fig.write_html("cluster_visualization.html")
print("聚类可视化已保存为 cluster_visualization.html")"""