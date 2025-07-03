import codecs
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SinglePassClustering:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.clusters = []  # 每个簇保存的是向量的平均表示
        self.cluster_contents = []  # 保存每个簇的原始文本索引

    def fit(self, vectors):
        is_sparse = hasattr(vectors, "toarray")

        for idx, vec in enumerate(vectors):
            if is_sparse:
                vec = vec.toarray().reshape(1, -1)  # 将稀疏矩阵转为密集矩阵
            else:
                vec = vec.reshape(1, -1)

            if not self.clusters:
                self.clusters.append(vec)
                self.cluster_contents.append([idx])
                continue

            similarities = [cosine_similarity(vec, c_vec)[0][0] for c_vec in self.clusters]
            max_sim = max(similarities)
            max_index = similarities.index(max_sim)

            if max_sim >= self.threshold:
                # 加入最近簇，并更新中心
                self.cluster_contents[max_index].append(idx)

                # 获取该簇中所有向量
                cluster_indices = self.cluster_contents[max_index]
                if is_sparse:
                    # 对稀疏矩阵，先提取簇中的向量，再转为密集矩阵
                    old_vecs = vectors[cluster_indices].toarray()
                else:
                    old_vecs = vectors[cluster_indices]

                # 计算新的簇中心
                self.clusters[max_index] = np.mean(old_vecs, axis=0).reshape(1, -1)
            else:
                self.clusters.append(vec)
                self.cluster_contents.append([idx])

        return self.cluster_contents


def cluster_csv_content(file_path, threshold=0.5):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在")
        print(f"当前工作目录: {os.getcwd()}")
        return None

    df = pd.read_csv(file_path)

    if 'content' not in df.columns:
        raise ValueError("CSV文件中必须包含 'content' 列。")

    contents = df['content'].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(contents)

    model = SinglePassClustering(threshold=threshold)
    clusters = model.fit(tfidf_matrix)

    # 打印聚类结果
    for idx, cluster in enumerate(clusters):
        print(f"--- 第 {idx + 1} 个簇（共{len(cluster)}条）---")
        for doc_index in cluster:
            print(f" - {contents[doc_index]}")
        print()

    return clusters


def generate_html_report(clusters, contents, file_path, threshold):
    """
    生成HTML聚类报告
    """
    # 使用 {{ 和 }} 来转义大括号，这样 Python 不会将其视为格式化占位符
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>文本聚类结果</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .summary {{
                background-color: #e9f7fe;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .cluster {{
                background-color: white;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .cluster-header {{
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px 5px 0 0;
                margin: -15px -15px 15px -15px;
            }}
            .document {{
                border-bottom: 1px solid #eee;
                padding: 10px 0;
            }}
            .document:last-child {{
                border-bottom: none;
            }}
        </style>
    </head>
    <body>
        <h1>文本聚类结果</h1>
        <div class="summary">
            <p><strong>聚类阈值:</strong> {threshold}</p>
            <p><strong>总文档数:</strong> {total_docs}</p>
            <p><strong>总簇数:</strong> {total_clusters}</p>
            <p><strong>最大簇大小:</strong> {max_cluster_size}</p>
            <p><strong>最小簇大小:</strong> {min_cluster_size}</p>
            <p><strong>平均簇大小:</strong> {avg_cluster_size:.2f}</p>
        </div>
    """.format(
        threshold=threshold,
        total_docs=len(contents),
        total_clusters=len(clusters),
        max_cluster_size=max([len(c) for c in clusters]) if clusters else 0,
        min_cluster_size=min([len(c) for c in clusters]) if clusters else 0,
        avg_cluster_size=sum([len(c) for c in clusters]) / len(clusters) if clusters else 0
    )

    # 添加每个簇的详细信息
    for idx, cluster in enumerate(clusters):
        html += """
        <div class="cluster">
            <div class="cluster-header">
                <h2>簇 #{num} (包含 {size} 条文档)</h2>
            </div>
        """.format(num=idx + 1, size=len(cluster))

        for doc_idx in cluster:
            html += """
            <div class="document">
                <p>{content}</p>
            </div>
            """.format(content=contents[doc_idx])

        html += "</div>"

    html += """
    </body>
    </html>
    """

    # 生成HTML文件
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write(html)

    print(f"聚类报告已保存到：{file_path}")
def generate_text_report(clusters, contents, file_path, threshold):
    """
    生成文本格式的聚类报告
    """
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write("文本聚类结果\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"聚类阈值: {threshold}\n")
        f.write(f"总文档数: {len(contents)}\n")
        f.write(f"总簇数: {len(clusters)}\n")

        if clusters:
            f.write(f"最大簇大小: {max([len(c) for c in clusters])}\n")
            f.write(f"最小簇大小: {min([len(c) for c in clusters])}\n")
            f.write(f"平均簇大小: {sum([len(c) for c in clusters]) / len(clusters):.2f}\n\n")

        # 添加每个簇的详细信息
        for idx, cluster in enumerate(clusters):
            f.write(f"\n簇 #{idx + 1} (包含 {len(cluster)} 条文档)\n")
            f.write("-" * 50 + "\n")

            for doc_idx in cluster:
                f.write(f"{contents[doc_idx]}\n")
                f.write("-" * 30 + "\n")

            f.write("\n")

    print(f"聚类文本报告已保存到：{file_path}")


def cluster_csv_content(file_path, threshold=0.5, output_dir="output"):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在")
        print(f"当前工作目录: {os.getcwd()}")
        return None

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    df = pd.read_csv(file_path)

    if 'content' not in df.columns:
        raise ValueError("CSV文件中必须包含 'content' 列。")

    contents = df['content'].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(contents)

    model = SinglePassClustering(threshold=threshold)
    clusters = model.fit(tfidf_matrix)

    # 打印聚类结果
    for idx, cluster in enumerate(clusters):
        print(f"--- 第 {idx + 1} 个簇（共{len(cluster)}条）---")
        for doc_index in cluster[:3]:  # 只打印前三条，避免输出过多
            print(f" - {contents[doc_index][:100]}...")  # 只打印前100个字符
        if len(cluster) > 3:
            print(f"   ... 等 {len(cluster) - 3} 条")
        print()

    # 获取原始文件名（不含路径和扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 生成HTML报告
    html_path = os.path.join(output_dir, f"{file_name}_clusters.html")
    generate_html_report(clusters, contents, html_path, threshold)

    # 生成文本报告
    text_path = os.path.join(output_dir, f"{file_name}_clusters.txt")
    generate_text_report(clusters, contents, text_path, threshold)

    return clusters


if __name__ == '__main__':
    # 使用绝对路径或正确的相对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'corpus', '拉非兹败选话题分析', 'oversea.csv')

    # 输出目录
    output_dir = os.path.join(base_dir, 'output')

    print(f"尝试读取文件: {csv_path}")
    print(f"输出目录: {output_dir}")

    cluster_csv_content(csv_path, threshold=0.4, output_dir=output_dir)  # 可调整阈值