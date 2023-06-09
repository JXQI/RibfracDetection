import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig= plt.figure()
ax= fig.add_subplot(111, projection='3d')

all_points = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
# 随机生成100个点
for i in range(100):
    generateddata = [random.randint(1, 100) / 100, random.randint(1, 100) / 100, random.randint(1, 100) / 100]
    if not generateddata in all_points:  # 去掉重复数据
        all_points.append(generateddata)

# 调用KMeans方法, 聚类数为4个，fit()之后开始聚类
kmeans = KMeans(n_clusters=4).fit(all_points)
# # 调用DBSCAN方法, eps为最小距离，min_samples 为一个簇中最少的个数，fit()之后开始聚类
# dbscan = DBSCAN(eps=0.132, min_samples=2).fit(all_points)


# plt.subplot(1, 2, 1)
# plt.title('kmeans')
for i, l in enumerate(kmeans.labels_):
    ax.scatter(all_points[i][0], all_points[i][1], all_points[i][2], s=20, c=None, depthshade=True,marker=markers[l],color=colors[l])
    # 开始画图
    # plt.plot(all_points[i][0], all_points[i][1], color=colors[l], marker=markers[l])

# plt.subplot(1, 2, 2)
# plt.title('dbscan')
# for i, l in enumerate(dbscan.labels_):
#     # if l == -1
#     plt.plot(all_points[i][0], all_points[i][1], color=colors[l], marker=markers[l])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
