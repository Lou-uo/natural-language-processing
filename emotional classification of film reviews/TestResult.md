模型训练完成！已生成以下结果：

**测试结果摘要：**

- 测试集准确率：**74.79%**
- AUC值：**0.8222**
- 平均精确率：**0.8192**

**分类报告：**
| 类别 | 精确率 | 召回率 | F1分数 |
| ---- | ------ | ------ | ------ |
| 负面 | 0.75   | 0.75   | 0.75   |
| 正面 | 0.75   | 0.75   | 0.75   |

**生成的图表：**
1. `class_distribution.png` - 数据集类别分布
2. `test_confusion_matrix.png` - 测试集混淆矩阵
3. `test_roc_curve.png` - ROC曲线
4. `test_pr_curve.png` - 精确率-召回率曲线

**示例预测：**
- "this movie is fantastic and amazing" → 正面 (置信度: 0.70)
- "terrible film, waste of time" → 负面 (置信度: 0.81)
- "the plot was boring and the acting was poor" → 负面 (置信度: 0.93)
- "one of the best movies i have ever seen" → 正面 (置信度: 0.82)

------

辅助工具：OpenCode