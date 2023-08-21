import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import utils
from sklearn.cluster import KMeans
# from cloudprior import CloudCluster
import seaborn as sns


class CloudCluster:
    def __init__(self, cpoint, nbs, ids, k=0, sim2c=None):
        self.k = k
        self.point_num = len(ids)
        self.points = ids
        self.center = cpoint
        self.c_neighbors = nbs
        self.sim2c = sim2c


def get_subsemantic_clouds(hf_clu, cloud, size=10):
    X = []
    sizes = []
    for clu in hf_clu:
        clu_cloud = []
        nbs_size = len(clu.points) if len(clu.points) < size else size
        for i in range(nbs_size):
            clu_cloud.append(cloud[clu.points[i]])
        X += clu_cloud
        sizes.append(nbs_size)
    pca = PCA(n_components=2)
    pca = pca.fit(X)
    X_dr = pca.transform(X)
    subsemantic_clouds = []
    startid = 0
    for sz in sizes:
        subsemantic_clouds.append(X_dr[startid:startid + sz])
        startid = startid + sz
    # self.X_dr_se_cloud = X_dr
    # self.subsemantic_clouds = subsemantic_clouds
    return subsemantic_clouds, X_dr


def prepare_semantic_tree_graph(hf_clu, cloud, clu_size=10, k=2):
    pca = PCA(n_components=2)
    pca = pca.fit(cloud)
    cloud_dr = pca.transform(cloud)
    cloud_ = cloud_dr
    subsemantic_clouds, X_dr = get_subsemantic_clouds(hf_clu, cloud, size=clu_size)

    c_ = np.mean(X_dr, axis=0)
    subse_cloud_ = subsemantic_clouds
    subse_ = [np.mean(clu_cloud_, axis=0) for clu_cloud_ in subsemantic_clouds]
    subse_nbs_ = []
    for clu_cloud_ in subsemantic_clouds:
        kmeans = KMeans(n_clusters=k).fit(clu_cloud_)
        subse_nbs_.append(kmeans.cluster_centers_)
    subse_nbs = []
    for clu in hf_clu:
        nbs = clu.c_neighbors[0:k]
        subse_nbs.append(nbs)

    stree = SemanticTree(c_, subse_, subse_nbs_, subse_cloud_, X_dr, cloud_, subse_nbs, k, clu_size)
    return stree


class SemanticTree:
    def __init__(self, c_, subse_:list, subse_nbs_:list, subse_cloud_:list,
                 se_cloud_, cloud_, subse_nbs:list, k, clu_size):
        self.c_ = c_
        self.subse_ = subse_
        self.subse_nbs_ = subse_nbs_
        self.subse_cloud_ = subse_cloud_
        self.se_cloud_ = se_cloud_
        self.cloud_ = cloud_
        self.subse_nbs = subse_nbs
        self.k = k
        self.clu_size = clu_size


def load_semantic_tree(k, clu_size):
    cloud = utils.load_from_disk('./pltdata/en/mouse/cloud2')
    hf_semantics = utils.load_from_disk('./pltdata/en/mouse/k12_cloud_hfsc_t2.pkl')
    stree = prepare_semantic_tree_graph(hf_semantics, cloud, clu_size=clu_size, k=k)

    cloud = utils.load_from_disk('./pltdata/en/mouse/cloud1')
    hf_semantics = utils.load_from_disk('./pltdata/en/mouse/k12_cloud_hfsc_t1.pkl')
    stree2 = prepare_semantic_tree_graph(hf_semantics, cloud, clu_size=clu_size, k=k)

    return stree, stree2


def plot_3dsurface():
    # 绘制面
    fig3 = plt.figure(3)
    ax3 = plt.subplot(projection='3d')
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, 50)
    ax3.set_zlim(0, 50)
    x = np.arange(1, 50, 1)
    y = np.arange(1, 50, 1)
    X, Y = np.meshgrid(x, y)  # 将坐标向量(x,y)变为坐标矩阵(X,Y)

    # def Z(X, Y):
    #     return X * 0.2 + Y * 0.3 + 20
    def Z(X, Y):
        return X * 0 + Y * 0 + 20

    Zt = Z(X, Y)
    ax3.plot_surface(X=Z(X, Y), Y=Y, Z=X, rstride=10, cstride=10, antialiased=True, color=(0.1, 0.2, 0.5, 0.3))
    ax3.set_xlabel('x轴')
    ax3.set_ylabel('y轴')
    ax3.set_zlabel('z轴')

    x = [30, 40, 50]
    ax3.scatter(x[0], x[1], x[2], s=20, color='b', marker='*')

    x = [20, 30, 30]
    y = [20, 30, 40]
    z = [20, 30, 50]
    ax3.plot3D(xs=x, ys=y, zs=z, color='blue')

    ax3.text(x=20, y=20, z=20, s="xxx", )

    ax3.grid(False)#不显示3d背景网格
    ax3.set_xticks([])#不显示x坐标轴
    ax3.set_yticks([])#不显示y坐标轴
    ax3.set_zticks([])#不显示z坐标轴
    plt.axis('off')#关闭所有坐标轴

    plt.title('三维曲面')

    plt.show()


# plot_3dsurface()

def max_clu_line_len(sub_nbs_c_, clu_center):
    max_l = 0
    for i in range(sub_nbs_c_.shape[0]):
        point = sub_nbs_c_[i, :]
        l = np.linalg.norm(point - clu_center)
        max_l = l if max_l < l else max_l
    return max_l


def nbs_point_position_assign(point, clu_center, max_l):
    while np.linalg.norm(point - clu_center) < max_l/3:
        point = 2 * (point - clu_center) + clu_center

    return point


def plot_semantic_tree(stree, t, ax3, X, Y, word, colors, bg_alpha=0.1, labeled=False):
    def Z(X, Y):
        return X * 0 + Y * 0 + t

    ax3.plot_surface(X=Z(X, Y), Y=Y, Z=X, rstride=10, cstride=10, antialiased=True, color=(0.1, 0.2, 0.5, bg_alpha))

    # draw center
    c_ = stree.c_
    if labeled:
        ax3.scatter(t, c_[0], c_[1], s=50, color='r', marker='o')
    else:
        ax3.scatter(t, c_[0], c_[1], s=50, color='r', marker='o', label='static embedding')
    ax3.text(t, c_[0], c_[1], s=word)
    # ax3.text(t, c_[0], c_[1], s='mouse')
    # ax3.text(t, c_[0], c_[1], s='Maus')
    # ax3.text(t, c_[0], c_[1], s='mus')

    line_colors = [colors[0], colors[1], colors[2]]
    markers_colors = [colors[5], colors[6], colors[7]]
    markers = ['o', '*', ]

    # show_cloud_num = 100 if 100 < stree.cloud_.shape[0] else stree.cloud_.shape[0]
    # for i in range(show_cloud_num):
    #     point = stree.cloud_[i, :]
    #     ax3.scatter(t, point[0], point[1], alpha=0.2, c='#778899', s=5, marker='o')

    for cid, clu_center in enumerate(stree.subse_):
        if len(stree.subse_) > 1:
            ax3.text(t, clu_center[0], clu_center[1], s=word+'_'+str(cid), alpha=0.9)
        if labeled:
            ax3.scatter(t, clu_center[0], clu_center[1], alpha=0.8, c='#C71585', s=30, marker='o')
        else:
            ax3.scatter(t, clu_center[0], clu_center[1], alpha=0.8, c='#C71585', s=30, marker='o',
                        label='polysemy embedding')
        sub_nbs_c_ = stree.subse_nbs_[cid]
        # plot line
        x = [t, t]
        y = [c_[0], clu_center[0]]
        z = [c_[1], clu_center[1]]
        ax3.plot3D(xs=x, ys=y, zs=z, color=line_colors[1], alpha=0.9)
        max_l = max_clu_line_len(sub_nbs_c_, clu_center)
        for i in range(sub_nbs_c_.shape[0]):
            point = sub_nbs_c_[i, :]
            point = nbs_point_position_assign(point, clu_center, max_l)
            if labeled:
                ax3.scatter(t, point[0], point[1], alpha=0.8, color='#1E90FF', s=22, marker='*')
            else:
                ax3.scatter(t, point[0], point[1], alpha=0.8, color='#1E90FF', s=22, marker='*',
                            label='neighbor embedding')
            labeled = True
            nb_w = stree.subse_nbs[cid][i]
            # temp for mark sc change branch
            # sc_wset = ['mousepad', 'keyboard', 'computer', 'Mouse', 'user']
            sc_wset = ['mousepad', 'keyboard', 'computer', 'Mouse', 'user']
            if nb_w in sc_wset:
                ax3.text(t, point[0]-0.3, point[1]+0.12, s=nb_w, c='darkorange')
            else:
                ax3.text(t, point[0]-0.3, point[1]+0.12, s=nb_w, alpha=0.7)
            # plot line
            x = [t, t]
            y = [point[0], clu_center[0]]
            z = [point[1], clu_center[1]]
            ax3.plot3D(xs=x, ys=y, zs=z, color=line_colors[2], alpha=0.9)
        # clu_cloud = stree.subse_cloud_[cid]
        # for i in range(clu_cloud.shape[0]):
        #     point = clu_cloud[i, :]
        #     ax3.scatter(t, point[0], point[1], alpha=0.4, c='#1E90FF', s=5, marker='o')

    # 绘制曲线绘制圆形螺旋曲线
    # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # z = np.linspace(-4, 4, 100) / 4
    # r = z ** 3 + 1
    # x = r * np.sin(theta)
    # y = r * np.cos(theta)
    # 绘制曲线绘制圆形
    # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # z = 3
    # r = 2
    # x = r * np.sin(theta)
    # y = r * np.cos(theta)

    # 绘制图形
    # ax3.plot(x, y, z, label='parametric curve')


def plot_3d_semantic_tree():
    # clu_size 是影响PCA选点的，不同的点集PCA结果不一样，可能是自由旋转的。
    word = 'mouse'
    stree, stree2 = load_semantic_tree(k=5, clu_size=20)
    print(stree.subse_nbs)
    print(stree2.subse_nbs)
    # return
    cloud = stree.se_cloud_
    cloud2 = stree2.se_cloud_
    cloud = np.concatenate((cloud, cloud2), axis=0)
    # draw plane
    ax_max = np.max(cloud)
    ax_min = np.min(cloud)
    lim_max = ax_max + (ax_max - ax_min) / 8
    lim_min = ax_min - (ax_max - ax_min) / 8
    # sns.set_style("whitegrid")
    colors = sns.color_palette('colorblind')

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 15,
    })

    fig3 = plt.figure(8, figsize=(10, 10))
    ax3 = plt.subplot(projection='3d')
    ax3.set_xlim(lim_min, lim_max)
    ax3.set_ylim(lim_min, lim_max)
    ax3.set_zlim(lim_min, lim_max)
    x = np.arange(lim_min, lim_max, (ax_max - ax_min) / 50)
    y = np.arange(lim_min, lim_max, (ax_max - ax_min) / 50)
    X, Y = np.meshgrid(x, y)  # 将坐标向量(x,y)变为坐标矩阵(X,Y)

    t1 = lim_min + (lim_max - lim_min) / 4
    plot_semantic_tree(stree2, t1, ax3, X, Y, word, colors)

    t2 = lim_min + ((lim_max - lim_min) / 4) * 4
    plot_semantic_tree(stree, t2, ax3, X, Y, word, colors, labeled=True)

    x = [t1, t2]
    y = [stree2.c_[0], stree.c_[0]]
    z = [stree2.c_[1], stree.c_[1]]
    ax3.plot3D(xs=x, ys=y, zs=z, color=colors[0], alpha=0.2)

    # ax3.text(t1, lim_min, lim_min, s='t1', fontsize=10,)
    # ax3.text(t2, lim_min, lim_min, s='t2', fontsize=10,)
    ax3.text(t1, lim_min, lim_min, s='t-1')
    ax3.text(t2, lim_min, lim_min, s='t')

    ax3.grid(False)#不显示3d背景网格
    ax3.set_xticks([])#不显示x坐标轴
    ax3.set_yticks([])#不显示y坐标轴
    ax3.set_zticks([])#不显示z坐标轴
    plt.axis('off')#关闭所有坐标轴

    # 调整初始显示角度
    ax3.view_init(elev=8, azim=-48)
    # ax3.view_init(elev=8, azim=-48, roll=0)
    # ax3.view_init(elev=0, azim=0, roll=0)

    # plt.title('Semantic Tree for Word \'mouse\'')
    # plt.legend(bbox_to_anchor=(0.95, 0.8))
    # plt.legend(frameon=False, loc=(0, 0.1), ncol=3, columnspacing=0.1, labelspacing=0.1, fontsize=14)
    plt.legend(frameon=False, loc=(0.05, 0.1), ncol=3, columnspacing=0, handletextpad=0, fontsize=15)
    # plt.legend(frameon=False, loc=(-0.2, -0.2), ncol=3, fontsize=16)
    save_path = './data/crossdiffusion/'
    plt.savefig(save_path+"mouse.pdf")
    plt.show()

    fig3 = plt.figure(3, figsize=(10, 10))
    ax_ = plt.subplot(projection='3d')
    ax_.grid(False)#不显示3d背景网格
    ax_.set_xticks([])#不显示x坐标轴
    ax_.set_yticks([])#不显示y坐标轴
    ax_.set_zticks([])#不显示z坐标轴
    plt.axis('off')#关闭所有坐标轴

    t2 = lim_min + ((lim_max - lim_min) / 4) * 4
    plot_semantic_tree(stree, t2, ax_, X, Y, word, colors, bg_alpha=0)
    # ax_.view_init(elev=0, azim=0, roll=0)
    ax_.view_init(elev=0, azim=0)
    # plt.legend(bbox_to_anchor=(0.8, 0.7))
    plt.legend(frameon=False, loc=(0.05, 0.26), ncol=3, columnspacing=0, handletextpad=0, fontsize=15)
    plt.savefig(save_path+"mouse-tree.pdf")
    plt.show()


plot_3d_semantic_tree()




