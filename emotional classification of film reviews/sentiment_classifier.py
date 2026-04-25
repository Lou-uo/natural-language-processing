import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(pos_file, neg_file):
    """加载正负样本数据"""
    with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
        pos_reviews = [line.strip() for line in f.readlines()]
    with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
        neg_reviews = [line.strip() for line in f.readlines()]

    # 创建标签：正面为1，负面为0
    pos_labels = [1] * len(pos_reviews)
    neg_labels = [0] * len(neg_reviews)

    # 合并数据
    reviews = pos_reviews + neg_reviews
    labels = pos_labels + neg_labels

    return reviews, labels

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title('混淆矩阵', fontsize=16)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return cm

def plot_roc_curve(y_true, y_prob, save_path='roc_curve.png'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)', fontsize=12)
    plt.ylabel('真正率 (TPR)', fontsize=12)
    plt.title('ROC曲线', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return roc_auc

def plot_precision_recall_curve(y_true, y_prob, save_path='pr_curve.png'):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {ap:.3f})')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线', fontsize=16)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return ap

def plot_class_distribution(y_train, y_val, y_test, save_path='class_distribution.png'):
    """绘制数据集分布情况"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    datasets = [('训练集', y_train), ('验证集', y_val), ('测试集', y_test)]
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for ax, (name, y), color in zip(axes, datasets, colors):
        unique, counts = np.unique(y, return_counts=True)
        labels = ['负面', '正面']
        ax.bar(labels, counts, color=[color, color])
        ax.set_title(f'{name}类别分布', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=10)
        for i, count in enumerate(counts):
            ax.text(i, count + 20, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=" * 60)
    print("电影评论情感分类模型")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    pos_file = 'rt-polarity.pos'
    neg_file = 'rt-polarity.neg'
    reviews, labels = load_data(pos_file, neg_file)
    print(f"数据加载完成：共 {len(reviews)} 条评论")
    print(f"  - 正面评论：{sum(labels)} 条")
    print(f"  - 负面评论：{len(labels) - sum(labels)} 条")

    # 2. 划分数据集：训练集0.8，验证集0.1，测试集0.1
    print("\n[2/6] 划分数据集...")
    # 先分出测试集（10%）
    X_temp, X_test, y_temp, y_test = train_test_split(
        reviews, labels, test_size=0.1, random_state=42, stratify=labels
    )
    # 从剩余数据中分出验证集（占原始的10%，即剩余数据的约11.11%）
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1/9, random_state=42, stratify=y_temp
    )

    print(f"  - 训练集：{len(X_train)} 条 ({len(X_train)/len(reviews)*100:.1f}%)")
    print(f"  - 验证集：{len(X_val)} 条 ({len(X_val)/len(reviews)*100:.1f}%)")
    print(f"  - 测试集：{len(X_test)} 条 ({len(X_test)/len(reviews)*100:.1f}%)")

    # 3. 可视化数据分布
    print("\n[3/6] 绘制数据分布图...")
    plot_class_distribution(y_train, y_val, y_test)

    # 4. 特征提取（TF-IDF）
    print("\n[4/6] 特征提取（TF-IDF）...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"特征维度：{X_train_tfidf.shape[1]}")

    # 5. 训练模型
    print("\n[5/6] 训练模型（逻辑回归）...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    model.fit(X_train_tfidf, y_train)
    print("模型训练完成！")

    # 6. 验证集评估
    print("\n[6/6] 模型评估...")
    val_pred = model.predict(X_val_tfidf)
    val_prob = model.predict_proba(X_val_tfidf)[:, 1]
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\n验证集准确率：{val_acc:.4f}")
    print("\n验证集分类报告：")
    print(classification_report(y_val, val_pred, target_names=['负面', '正面']))

    # 测试集评估
    test_pred = model.predict(X_test_tfidf)
    test_prob = model.predict_proba(X_test_tfidf)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\n{'='*60}")
    print(f"测试集准确率：{test_acc:.4f}")
    print(f"{'='*60}")
    print("\n测试集分类报告：")
    print(classification_report(y_test, test_pred, target_names=['负面', '正面']))

    # 绘制测试集混淆矩阵
    print("\n绘制测试集混淆矩阵...")
    cm = plot_confusion_matrix(y_test, test_pred, 'test_confusion_matrix.png')
    print(f"混淆矩阵已保存为 test_confusion_matrix.png")

    # 绘制ROC曲线
    print("\n绘制ROC曲线...")
    roc_auc = plot_roc_curve(y_test, test_prob, 'test_roc_curve.png')
    print(f"ROC曲线已保存为 test_roc_curve.png")
    print(f"AUC值：{roc_auc:.4f}")

    # 绘制精确率-召回率曲线
    print("\n绘制精确率-召回率曲线...")
    ap = plot_precision_recall_curve(y_test, test_prob, 'test_pr_curve.png')
    print(f"PR曲线已保存为 test_pr_curve.png")
    print(f"平均精确率：{ap:.4f}")

    # 示例预测
    print("\n" + "="*60)
    print("示例预测：")
    print("="*60)
    test_samples = [
        "this movie is fantastic and amazing",
        "terrible film, waste of time",
        "the plot was boring and the acting was poor",
        "one of the best movies i have ever seen"
    ]
    for sample in test_samples:
        sample_tfidf = vectorizer.transform([sample])
        pred = model.predict(sample_tfidf)[0]
        prob = model.predict_proba(sample_tfidf)[0]
        sentiment = "正面" if pred == 1 else "负面"
        print(f"  文本: {sample}")
        print(f"  预测: {sentiment} (置信度: {prob[pred]:.4f})\n")

    print("\n所有结果和图表已保存！")
    print("="*60)

if __name__ == "__main__":
    main()
