import codecs
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import datetime
from dateutil.parser import parse


# 读取停用词函数
def load_stopwords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        print(f"警告：未找到停用词文件 '{file_path}'")
        return set()


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


def format_time_range(time_range):
    """格式化时间范围显示"""
    if not time_range or len(time_range) != 2:
        return "时间范围未知"

    start_time, end_time = time_range

    # 尝试格式化时间，失败则返回原始值
    try:
        # 如果是时间戳（数值型）
        if isinstance(start_time, (int, float)):
            start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')
            end_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M')
        else:
            # 尝试解析字符串格式时间
            start_str = parse(str(start_time)).strftime('%Y-%m-%d %H:%M')
            end_str = parse(str(end_time)).strftime('%Y-%m-%d %H:%M')

        return f"{start_str} 至 {end_str}"
    except Exception as e:
        print(f"时间格式化错误: {e}")
        return f"{start_time} 至 {end_time}"


def generate_html_report(clusters, contents, time_data, file_path, threshold):
    """
    生成HTML聚类报告，包含时间分布信息
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
            .time-range {{
                background-color: #f2f2f2;
                padding: 8px;
                margin-bottom: 15px;
                border-radius: 3px;
                font-size: 0.9em;
                color: #555;
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
        # 获取该簇的时间范围
        time_range = None
        if time_data:
            cluster_times = [time_data[doc_idx] for doc_idx in cluster if time_data[doc_idx] is not None]
            if cluster_times:
                time_range = [min(cluster_times), max(cluster_times)]

        html += """
        <div class="cluster">
            <div class="cluster-header">
                <h2>簇 #{num} (包含 {size} 条文档)</h2>
            </div>
        """.format(num=idx + 1, size=len(cluster))

        # 添加时间范围信息
        if time_range:
            time_str = format_time_range(time_range)
            html += """
            <div class="time-range">
                <strong>时间范围:</strong> {time_range}
            </div>
            """.format(time_range=time_str)

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


def generate_text_report(clusters, contents, time_data, file_path, threshold):
    """
    生成文本格式的聚类报告，包含时间分布信息
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

            # 添加时间范围信息
            if time_data:
                cluster_times = [time_data[doc_idx] for doc_idx in cluster if time_data[doc_idx] is not None]
                if cluster_times:
                    time_range = [min(cluster_times), max(cluster_times)]
                    time_str = format_time_range(time_range)
                    f.write(f"时间范围: {time_str}\n")
                    f.write("-" * 30 + "\n")

            for doc_idx in cluster:
                f.write(f"{contents[doc_idx]}\n")
                f.write("-" * 30 + "\n")

            f.write("\n")

    print(f"聚类文本报告已保存到：{file_path}")


def parse_time_field(value):
    """解析各种可能的时间格式"""
    if pd.isna(value) or value is None:
        return None

    # 尝试解析时间戳（如果是数值）
    if isinstance(value, (int, float)):
        try:
            return value
        except:
            pass

    # 尝试解析字符串格式时间
    try:
        return parse(str(value))
    except:
        pass

    return None


def cluster_texts(file_path, content_field='content', time_field=None, threshold=0.5, output_dir="output",
                  max_docs=1000, use_transformer=True):
    """
    通用文本聚类函数，可处理CSV和JSONL文件
    增加了时间字段处理
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在")
        print(f"当前工作目录: {os.getcwd()}")
        return None

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 根据文件扩展名确定处理方式和时间字段
    file_extension = os.path.splitext(file_path)[1].lower()

    # 根据文件类型自动选择时间字段（如果未指定）
    if time_field is None:
        if file_extension == '.csv':
            time_field = 'utime'
        elif file_extension == '.jsonl':
            time_field = 'publish_time'

    print(f"使用时间字段: {time_field}")

    if file_extension == '.csv':
        print(f"检测到CSV文件: {file_path}")
        contents, time_data = read_csv_file(file_path, content_field, time_field, max_docs)
    elif file_extension == '.jsonl':
        print(f"检测到JSONL文件: {file_path}")
        contents, time_data = read_jsonl_file(file_path, content_field, time_field, max_docs)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}，仅支持.csv和.jsonl文件")

    if not contents:
        print("未能成功读取文本内容，请检查文件格式和内容字段名")
        return None

    # 加载多语言停用词
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stopwords = set()
    stopwords_files = [
        os.path.join(base_dir, 'stopwords', 'Arabic.txt'),
        os.path.join(base_dir, 'stopwords', 'Indonesian.txt'),
        os.path.join(base_dir, 'stopwords', 'Malay.txt'),
        os.path.join(base_dir, 'stopwords', 'English.txt'),
        os.path.join(base_dir, 'stopwords', 'Chinese.txt')
    ]

    for sw_file in stopwords_files:
        stopwords.update(load_stopwords(sw_file))
    print(f"加载了 {len(stopwords)} 个停用词")

    if use_transformer:
        # 使用多语言SentenceTransformer生成文本向量
        try:
            print("正在加载多语言模型 paraphrase-multilingual-MiniLM-L12-v2...")
            model_path = "./bertopic_multilingual_model"
            embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder=model_path)
            print("模型加载完成，开始生成文本嵌入向量...")

            # 生成文本嵌入向量
            vectors = embedding_model.encode(contents, show_progress_bar=True)
            print(f"成功生成 {len(vectors)} 条文本的嵌入向量")
        except Exception as e:
            print(f"加载SentenceTransformer模型失败: {str(e)}")
            print("退回到使用TF-IDF向量化...")
            use_transformer = False

    if not use_transformer:
        # 退回到TF-IDF方法
        print("使用TF-IDF向量化文本...")
        vectorizer = TfidfVectorizer(stop_words=list(stopwords) if stopwords else None)
        vectors = vectorizer.fit_transform(contents)
        print(f"成功生成TF-IDF向量矩阵，形状: {vectors.shape}")

    # 聚类
    print(f"开始使用阈值 {threshold} 进行单程聚类...")
    model = SinglePassClustering(threshold=threshold)
    clusters = model.fit(vectors)
    print(f"聚类完成，共生成 {len(clusters)} 个簇")

    # 打印聚类结果
    for idx, cluster in enumerate(clusters):
        print(f"--- 第 {idx + 1} 个簇（共{len(cluster)}条）---")

        # 打印时间范围
        if time_data:
            cluster_times = [time_data[doc_idx] for doc_idx in cluster if time_data[doc_idx] is not None]
            if cluster_times:
                time_range = [min(cluster_times), max(cluster_times)]
                print(f"时间范围: {format_time_range(time_range)}")

        # 打印文本示例
        for doc_index in cluster[:3]:  # 只打印前三条，避免输出过多
            print(f" - {contents[doc_index][:100]}...")  # 只打印前100个字符
        if len(cluster) > 3:
            print(f"   ... 等 {len(cluster) - 3} 条")
        print()

    # 获取原始文件名（不含路径和扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 生成HTML报告
    html_path = os.path.join(output_dir, f"{file_name}_clusters.html")
    generate_html_report(clusters, contents, time_data, html_path, threshold)

    # 生成文本报告
    text_path = os.path.join(output_dir, f"{file_name}_clusters.txt")
    generate_text_report(clusters, contents, time_data, text_path, threshold)

    return clusters


def read_csv_file(file_path, content_field='content', time_field=None, max_docs=1000):
    """读取CSV文件中的文本内容和时间字段"""
    try:
        df = pd.read_csv(file_path)
        if content_field not in df.columns:
            raise ValueError(f"CSV文件中必须包含 '{content_field}' 列。")

        # 限制处理的最大文档数
        if len(df) > max_docs:
            print(f"数据量超过限制，仅处理前 {max_docs} 条记录")
            df = df.iloc[:max_docs]

        contents = df[content_field].astype(str).tolist()
        print(f"已从CSV文件中读取 {len(contents)} 条文本")

        # 提取时间数据
        time_data = None
        if time_field and time_field in df.columns:
            time_data = [parse_time_field(t) for t in df[time_field].tolist()]
            valid_times = sum(1 for t in time_data if t is not None)
            print(f"成功解析 {valid_times}/{len(time_data)} 条时间数据")

        return contents, time_data
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None, None


def read_jsonl_file(file_path, content_field='content', time_field=None, max_docs=1000):
    """读取JSONL文件中的文本内容和时间字段"""
    contents = []
    time_data = [] if time_field else None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_docs:
                    print(f"数据量超过限制，仅处理前 {max_docs} 条记录")
                    break

                try:
                    data = json.loads(line)
                    if content_field in data:
                        contents.append(str(data[content_field]))

                        # 提取时间字段
                        if time_field:
                            if time_field in data:
                                time_data.append(parse_time_field(data[time_field]))
                            else:
                                time_data.append(None)
                    else:
                        print(f"警告: 在第 {i + 1} 行找不到字段 '{content_field}'")
                except json.JSONDecodeError:
                    print(f"警告: 无法解析第 {i + 1} 行的JSON数据")

        print(f"已从JSONL文件中读取 {len(contents)} 条文本")

        if time_data:
            valid_times = sum(1 for t in time_data if t is not None)
            print(f"成功解析 {valid_times}/{len(time_data)} 条时间数据")

        return contents, time_data
    except Exception as e:
        print(f"读取JSONL文件时出错: {str(e)}")
        return None, None


if __name__ == '__main__':
    # 设置参数
    THRESHOLD = 0.6  # 聚类阈值
    MAX_DOCS = 100000  # 最大处理文档数
    USE_TRANSFORMER = True  # 使用多语言Transformer模型
    CONTENT_FIELD = 'content'  # 内容字段名称
    TIME_FIELD = None  # 自动检测时间字段（csv用utime，jsonl用publish_time）

    # 使用绝对路径或正确的相对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 自动检测可用的输入文件
    input_files = []
    potential_paths = [
        os.path.join(base_dir, '数据清洗', 'twitter.csv'),
        os.path.join(base_dir, '数据清洗', 'twitter.jsonl'),
        os.path.join(base_dir, 'data', 'twitter.csv'),
        os.path.join(base_dir, 'data', 'twitter.jsonl')
    ]

    for path in potential_paths:
        if os.path.exists(path):
            input_files.append(path)

    # 输出目录
    output_dir = os.path.join(base_dir, 'output')

    if not input_files:
        print("错误: 未找到可处理的文件。请确保在以下路径存在CSV或JSONL文件:")
        for path in potential_paths:
            print(f" - {path}")
    else:
        for file_path in input_files:
            print("\n" + "=" * 50)
            print(f"开始处理文件: {file_path}")
            print("=" * 50)

            # 文件扩展名
            ext = os.path.splitext(file_path)[1].lower()
            print(f"文件类型: {ext}")
            print(f"输出目录: {output_dir}")
            print(f"使用聚类阈值: {THRESHOLD}")
            print(f"处理数据量上限: {MAX_DOCS}条")
            print(f"使用多语言模型: {'是' if USE_TRANSFORMER else '否'}")
            print(f"内容字段名称: {CONTENT_FIELD}")
            print("-" * 50)

            cluster_texts(
                file_path,
                content_field=CONTENT_FIELD,
                time_field=TIME_FIELD,  # 自动检测时间字段
                threshold=THRESHOLD,
                output_dir=output_dir,
                max_docs=MAX_DOCS,
                use_transformer=USE_TRANSFORMER
            )