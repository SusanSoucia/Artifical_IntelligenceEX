import graphviz as graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from sklearn import metrics
from sklearn import tree
import os


# 配置环境变量
os.environ["PATH"] += os.pathsep + r'D:\softwares\Graphviz\bin'
print("PATH: ", os.environ["PATH"])


# 可视化特征重要性并且降序表示
def plot_feature_importances(feature_importances, title, feature_names, normalize=True):
    if normalize:
        # 直接转换为百分比（总和为100%），更直观
        feature_importances = feature_importances * 100
    # 降序排序
    index_sorted = np.argsort(feature_importances)[::-1]  # 等价于 flipud
    pos = np.arange(index_sorted.shape[0]) + 0.5
    
    plt.figure(figsize=(12, 8))
    plt.bar(pos, feature_importances[index_sorted], align='center', color='skyblue')
    plt.xticks(pos, np.array(feature_names)[index_sorted], rotation=45, ha='right')
    plt.ylabel('Relative Importance (%)' if normalize else 'Importance')
    plt.title(title)
    
    # 在柱子上显示数值
    for a, b in zip(pos, feature_importances[index_sorted]):
        plt.text(a, b + (0.5 if normalize else 0.01), f'{b:.2f}%' if normalize else f'{b:.3f}',
                 ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    # ---------------------  下方是数据的导入和处理 ---------------------
    data = pd.read_csv(open('/home/susan/store_code/artificial/aiex4/penguin.csv'))
    # 查看数据信息
    # print(data.info())
    # 补全缺失值
    data = data.fillna(-1)
    # fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
    # value：固定值，可以用固定数字、均值、中位数、众数等，此外还可以用字典，series等形式数据；
    # method:填充方法，'bfill','backfill','pad','ffill'
    # axis: 填充方向，默认0和index，还可以填1和columns
    # inplace:在原有数据上直接修改
    # limit:填充个数，如1，每列只填充1个缺失值
    # print(data['Sex'].unique())
    # print(data['Species'].unique())
    # print(data['Island'].unique())

    # 将分类变量转化成数字变量，方便后续计算
    def trans(x):
        if x == data['Species'].unique()[0]:
            return 0  # Adelie
        if x == data['Species'].unique()[1]:
            return 1  # Gentoo
        if x == data['Species'].unique()[2]:
            return 2  # Chinstrap
        if x == data['Island'].unique()[0]:
            return 0  # Torgersen
        if x == data['Island'].unique()[1]:
            return 1  # Biscoe
        if x == data['Island'].unique()[2]:
            return 2  # Dream
        if x == data['Sex'].unique()[0]:
            return 0  # male
        if x == data['Sex'].unique()[1]:
            return 1  # female
        if x == data['Sex'].unique()[2]:
            return -1  # -1

    # 将类型变量转换为值变量
    data['Species'] = data['Species'].apply(trans)
    data['Island'] = data['Island'].apply(trans)
    data['Sex'] = data['Sex'].apply(trans)
    # ---------------------  上方是数据的导入和处理 ---------------------

    feature_data = data[
        ['Island', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex',
         'Age']]
    goal_data = data[['Species']]

    # ---------------------  决策树训练与测试 ---------------------
    # 划分训练集 测试集

    x_train, x_test, y_train, y_test = train_test_split(feature_data, goal_data, test_size=0.2, random_state=2022)

    # 超参数学习曲线 这里只画了一个层数的
    test = []
    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                          max_depth=i + 1,
                                          random_state=2020,
                                          # 最大深度
                                          splitter='best'
                                          )  # 生成决策树分类器   entropy

        clf = clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        test.append(score)

    plt.plot(range(1, 11), test, color='red')
    plt.ylabel('score')
    plt.xlabel('max_depth')
    # plt.show()

    max_depth = test.index(max(test)) + 1
    print("该决策树的最佳层数是：", max_depth)

    # 训练决策树
    penguin_tree = DecisionTreeClassifier(criterion='entropy',
                                          splitter='best',
                                          random_state=2022,
                                          max_depth=max_depth)
    penguin_tree.fit(x_train, y_train)

    # 返回预测的准确度
    print('训练集预测成功率:', penguin_tree.score(x_train, y_train))
    print("********** 决策树 ***************")
    print('测试集预测成功率: %.3f' % penguin_tree.score(x_test, y_test))


    # ********** KNN ***************
    print("********** KNN ***************")
    # 定义不同的K值和距离度量方法
    k_values = [i for i in range(1, 20, 2)]  # 你可以根据需要添加更多的K值
    distance_metrics = ['euclidean', 'manhattan', 'minkowski']

    # 存储最佳性能的K值和距离度量方法
    best_k = None
    best_metric = None
    best_accuracy = 0

    # 遍历不同的K值和距离度量方法
    for k in k_values:
        for metric in distance_metrics:
            # 创建KNN分类器实例
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

            # 训练KNN分类器
            knn.fit(x_train, y_train.values.ravel())

            # 使用训练好的分类器对测试集进行预测
            y_pred = knn.predict(x_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)

            # 打印当前K值和距离度量方法的性能
            print(f"K = {k}, Metric = {metric}, 测试集预测成功率: {accuracy:.3f}")

            # 检查是否是目前为止最好的性能
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_metric = metric

    # 打印最佳性能的K值和距离度量方法
    print(f"Best K: {best_k}, Best Metric: {best_metric}, Best Accuracy: {best_accuracy:.3f}")
    # ********** KNN ***************


    print("********** 决策树 ***************")
    print('测试集预测成功率: %.3f' % penguin_tree.score(x_test, y_test))
    print("********** KNN ***************")
    print(f"Best K: {best_k}, Best Metric: {best_metric}, Best Accuracy: {best_accuracy:.3f}")


    # 画决策树
    feature_names = ['Island', 'Culmen Length (mm)', 'Culmen Depth (mm)',
                     'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Age']
    target_names = ['Adelie', 'Gentoo', 'Chinstrap']

    print("决策树各特征重要性（百分比）：")
    for name, imp in zip(feature_names, penguin_tree.feature_importances_):
        print(f"{name}: {imp*100:.2f}%")

    plot_feature_importances(penguin_tree.feature_importances_, 
                            'Penguin Species Decision Tree - Feature Importance',
                            feature_names,
                            normalize=True)

    dot_data = tree.export_graphviz(penguin_tree,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    out_file=None,
                                    filled=True)

    graph = graphviz.Source(dot_data)

    graph.render("penguin_tree-ex4")

    # # 在训练集和测试集上分布利用训练好的模型进行预测
    # train_predict = penguin_tree.predict(x_train)
    # test_predict = penguin_tree.predict(x_test)
    #
    # ## 利用accuracy 【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    # print('训练集预测成功率:', metrics.accuracy_score(y_train, train_predict))
    # print('测试集预测成功率:', metrics.accuracy_score(y_test, test_predict))
    # ---------------------  决策树训练与测试 ---------------------

    # --------------------- 结果可视化 ---------------------
    # 查看混淆矩阵，用训练好的决策树处理测试集
    confusion_matrix = metrics.confusion_matrix(penguin_tree.predict(x_test), y_test)
    plt.figure()
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('./matrix-ex4.pdf')
    plt.show()
    # --------------------- 结果可视化 ---------------------

    print(feature_data.corr())
    # 或用heatmap可视化
    sns.heatmap(feature_data.corr(), annot=True, cmap='coolwarm')
    plt.show()

    


if __name__ == '__main__':
    main()

