import utils
from tqdm import tqdm
import numpy as np
import rundata as rd
import treesim as ts
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment
import semeval_task2 as st2
from utils import clear_sim_words
# from main import evaluate_task1


class CloudCluster:
    def __init__(self, cpoint, nbs, ids, k=0, sim2c=None):
        self.k = k
        self.point_num = len(ids)
        self.points = ids
        self.center = cpoint
        self.c_neighbors = nbs
        self.sim2c = sim2c


def embed_filter(static_embedding):
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    number_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index_voc, voc_index, embeddings = static_embedding
    filter_out = ''


# def has_num(w):
#     number_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     for num in number_tokens:
#         if num in w:
#             return True
#     return False
#
#
# def clear_sim_words(keyword, embed, simsize, static_embeds):
#     index_voc, voc_index, embeddings = static_embeds
#     special_tokens = [keyword, "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
#
#     stop_words = utils.stop_words(language="en")
#     sizegap = 20
#     words = utils.most_sim_words(embed, simsize + sizegap, voc_index, embeddings, willprint=False)
#     poly_sims = []
#     for i, w in enumerate(words):
#         # print('id: {} w: {} sim: {}'.format(i, w[0], w[1]))
#         if w[0] in special_tokens:
#             continue
#         if w[0] in stop_words:
#             continue
#         if len(w[0]) < 2:
#             continue
#         if has_num(w[0]):
#             continue
#         if len(poly_sims) < simsize:
#             poly_sims.append(w[0])
#
#     return poly_sims


def eva_polysim_linear_assignment(polyset1, polyset2, static_embeds):
    index_voc, voc_index, embeddings = static_embeds
    weights = np.zeros((len(polyset1), len(polyset2)))
    for i, w_row in enumerate(polyset1):
        for j, w_col in enumerate(polyset2):
            weight = utils.word_cosine_similarity(w_row, w_col, voc_index, embeddings)
            weights[i, j] = weight
    row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
    mean = weights[row_ind, col_ind].mean()
    # sum = weights[row_ind, col_ind].sum()
    # print("keyword weights : ", weights)
    # print("keyword weights sum {} mean {}", sum, mean)
    # print("linear_assignment match weights max {} min {}".
    #       format(weights[row_ind, col_ind].max(), weights[row_ind, col_ind].min()))

    return mean


def neighbor_similarity(smset1, smset2, static_embeds):
    polytree_sim = eva_polysim_linear_assignment(smset1, smset2, static_embeds)
    return polytree_sim


def cloud_distance(cloud, static_embeds, keyword, k=12):
    # index_voc, voc_index, embeddings = static_embeds
    nbs_cloud = []
    sim_matrix = np.ones((len(cloud), len(cloud)), dtype='float32')
    for i, point in enumerate(cloud):
        print("{} itr {}".format(keyword, i))
        nbs = clear_sim_words(keyword, point, k, static_embeds)
        for j, nbs_ in enumerate(nbs_cloud):
            sim = neighbor_similarity(nbs, nbs_, static_embeds)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
        nbs_cloud.append(nbs)
    return sim_matrix, nbs_cloud


def cloud_distance_mp_unit(pairs, nbs_cloud, static_embeds, pid, return_dict):
    print(" cloud distance mp unit starting pid {} pairs num {}".format(pid, len(pairs)))
    sim_cloud = {}
    for pair in tqdm(pairs):
        # if i % 200 == 0:
        #     print(" process id {} progress {} %".format(pid, i/len(pairs)*100))
        nbs = nbs_cloud[pair[0]]
        nbs_ = nbs_cloud[pair[1]]
        sim = neighbor_similarity(nbs, nbs_, static_embeds)
        sim_cloud[pair] = sim
    return_dict[pid] = sim_cloud


def cloud_distance_mp(cloud, static_embeds, keyword, k=12, p_number=8):
    # index_voc, voc_index, embeddings = static_embeds
    nbs_cloud = []
    pairs = []
    sim_matrix = np.ones((len(cloud), len(cloud)), dtype='float32')
    clouds = []
    for i, point in enumerate(cloud):
        clouds.append((i, point))
    for point in tqdm(clouds):
        i, emb = point
        nbs = clear_sim_words(keyword, emb, k, static_embeds)
        for j, nbs_ in enumerate(nbs_cloud):
            pairs.append((i, j))
        nbs_cloud.append(nbs)

    manager = mp.Manager()
    return_dict = manager.dict()

    print(" Total pairs ", len(pairs))
    procs = []
    step = len(pairs) // p_number
    for pid in range(p_number):
        sub_pairs = pairs[pid*step: (pid+1)*step] if pid < p_number-1 else pairs[pid*step:]
        p = mp.Process(target=cloud_distance_mp_unit, args=(
            sub_pairs, nbs_cloud, static_embeds, pid, return_dict))
        p.start()
        procs.append(p)
        print("start process id ", pid)

    for p in procs:
        p.join()
    for pid in range(p_number):
        # print("return dict pid {} dict {}".format(pid, return_dict[pid]))
        pairs_dict = return_dict[pid]
        for pair in pairs_dict:
            sim_matrix[pair[0], pair[1]] = pairs_dict[pair]
            sim_matrix[pair[1], pair[0]] = pairs_dict[pair]
    return sim_matrix, nbs_cloud


def gen_cloud_distance(cloud, static_embeds, keyword, language, time, k=12, save_nbs=True):
    print(" cloud points ", len(cloud))
    matrix_path = './data/{}/words/{}/cloud_sim_matrix/k{}_cloud_matrix_t{}.npy'.format(language, keyword, k, time)
    nbs_path = './data/{}/words/{}/cloud_sim_matrix/k{}_cloud_nbs_t{}.pkl'.format(language, keyword, k, time)
    if utils.exists(matrix_path):
        print(" cloud points distance matrix exists ")
        return load_cloud_prior(matrix_path, nbs_path, load_nbs=save_nbs)

    sim_matrix, nbs_cloud = cloud_distance_mp(cloud, static_embeds, keyword, k=k)
    utils.create_filepath_dir(matrix_path)
    np.save(matrix_path, sim_matrix)
    # ts.plt_weights(sim_matrix)
    if save_nbs:
        utils.save_to_disk(nbs_path, nbs_cloud)
    return sim_matrix, nbs_cloud


def load_cloud_prior(matrix_path, nbs_path=None, load_nbs=True):
    print(" load cloud distance matrix prior ")
    # matrix_path = './data/{}/words/{}/cloud_sim_matrix/k{}_cloud_matrix_t{}.npy'.format(language, keyword, k, time)
    # nbs_path = './data/{}/words/{}/cloud_sim_matrix/k{}_cloud_nbs_t{}.pkl'.format(language, keyword, k, time)
    utils.create_filepath_dir(matrix_path)
    sim_matrix = np.load(matrix_path)

    nbs_cloud = None
    if load_nbs:
        nbs_cloud = utils.load_from_disk(nbs_path)

    return sim_matrix, nbs_cloud


def get_cloud_data(run_data):
    keyword, cloud1, cloud2, static_embeds1, static_embeds2, static_embeds, separate = run_data
    print("Runtime get cloud data, separate is ", separate)
    return keyword, cloud1, cloud2, static_embeds1, static_embeds2, static_embeds


def run_cloud(keywords, pmodel, corpus1, corpus2, language,
              lan_emb=None, k=12, separate=False, rt=0.73, sct=0.60, evasc=False):
    semantic_change_hf = {}
    semantic_change_hf_lf = {}
    semantic_change_grade = {}
    f = rd.gen_run_data(keywords, pmodel, corpus1, corpus2, language, lan_emb=lan_emb, separate=separate)
    for run_data in f:
        keyword, cloud1, cloud2, static_embeds1, static_embeds2, static_embeds = \
            get_cloud_data(run_data)
        hf_semantic1, lf_semantic1 = \
            cloud_cluster(cloud1, static_embeds1, keyword, language,
                          time='1', k=k, rt=rt, sct=sct)
        hf_semantic2, lf_semantic2 = \
            cloud_cluster(cloud2, static_embeds2, keyword, language,
                          time='2', k=k, rt=rt, sct=sct)
        # evaluate only in hf_semantics
        sc, grade = semantic_change_cloud_prior(hf_semantic1, lf_semantic1, hf_semantic2, lf_semantic2,
                                                static_embeds, rt=0.99, sct=sct)
        semantic_change_hf[keyword] = sc
        print(" @@@@@@@@@@@@@@@@@@ {} sc {} ".format(keyword, sc))

        # evaluate both in hf_semantics and lf_semantics
        sc, _ = semantic_change_cloud_prior(hf_semantic1, lf_semantic1, hf_semantic2, lf_semantic2,
                                            static_embeds, rt=rt, sct=sct)
        semantic_change_hf_lf[keyword] = sc
        semantic_change_grade[keyword] = grade
    if not evasc:
        return
    sc_results_path = './data/{}/semeval_task/sc_b/sct{}/rt{}/semantic_change_hf.pkl'.format(language, sct, rt)
    utils.save_to_disk(sc_results_path, semantic_change_hf)
    sc_results_path = './data/{}/semeval_task/sc_b/sct{}/rt{}/semantic_change_hf_lf.pkl'.format(language, sct, rt)
    utils.save_to_disk(sc_results_path, semantic_change_hf_lf)
    print("Result semantic_change_hf", semantic_change_hf)
    print("Result semantic_change_hf_lf", semantic_change_hf_lf)

    st2.evaluate_semeval_task2(semantic_change_grade, language)

    return semantic_change_hf, semantic_change_hf_lf


def semeval_cloud_distance():
    # keywords = utils.load_from_disk('./corpus/en/semevalSC_r/keywords')
    # print(keywords)
    # keywords = ["attack", "plane"]
    # keywords = ["attack", "plane", "edge", "stab", "tip", "thump"]
    # keywords = ["bit", "attack", "ball", "risk"]
    # keywords = ["stroke", "ounce"]
    keywords = ["bit"]
    pmodel = 'bert-base-multilingual-cased'  # 'bert-base-uncased'
    corpus1 = './corpus/en/semevalSC_r/data_ccoha1'
    corpus2 = './corpus/en/semevalSC_r/data_ccoha2'
    # corpus1 = './data/en/semeval/data_ccoha1'
    # corpus2 = './data/en/wiki/data_p1p41242'
    language = 'en'
    static_embedding = './data/en/static_embed.pkl'  # './data/multi/static_embed.pkl'
    # run_cloud(keywords, pmodel, corpus1, corpus2, language,
    #           lan_emb=None, k=12, separate=None, rt=0.73, sct=0.60)

    # keywords = ["Maus"]
    # language = 'de'
    # corpus1 = './corpus/de/semeval/data_corpus1'
    # corpus2 = './corpus/de/wiki/data_p1p297012'
    # run_cloud(keywords, pmodel, corpus1, corpus2, language,
    #           lan_emb=None, k=12, separate=False, rt=0.8, sct=0.8)

    keywords = ["mus"]
    language = 'sv'
    corpus1 = './corpus/sv/semeval/data_corpus1'
    corpus2 = './corpus/sv/wiki/data_svp153416p666977'
    run_cloud(keywords, pmodel, corpus1, corpus2, language,
              lan_emb=None, k=12, separate=False, rt=0.8, sct=0.8)


def points_set_distances(clusters1, clusters2, sim_matrix, static_embeds=None):
    points1 = clusters1.points
    points2 = clusters2.points
    if len(points1)*len(points2) < 1000 or static_embeds is None:
        sims = np.zeros((len(points1), len(points2)), dtype='float32')
        for m, i in enumerate(points1):
            for n, j in enumerate(points2):
                sims[m, n] = sim_matrix[i, j]
        return sims.flatten().mean()
    else:
        # print("len(points1)*len(points2) > 1000 is ", len(points1)*len(points2))
        return neighbor_similarity(clusters1.c_neighbors, clusters2.c_neighbors, static_embeds)


def clusters_distances(clusters, sim_matrix, rip_dist=None, static_embeds=None):
    # print(" clusters_distances start")
    cluster_num = len(clusters)
    cluster_dist = np.zeros((cluster_num, cluster_num), dtype='float32')
    if rip_dist is not None:
        cluster_dist[:rip_dist.shape[0], :rip_dist.shape[1]] = rip_dist
        i = cluster_num - 1
        for j in range(cluster_num-1):
            # i_points = clusters[i].points
            # j_points = clusters[j].points
            csim = points_set_distances(clusters[i], clusters[j], sim_matrix, static_embeds)
            cluster_dist[i, j] = csim
            cluster_dist[j, i] = csim
        # ts.plt_weights(cluster_dist)
        # print(" clusters_distances end")
        return cluster_dist
    else:
        for i in range(cluster_num):
            for j in range(i + 1, cluster_num):
                # i_points = clusters[i].points
                # j_points = clusters[j].points
                csim = points_set_distances(clusters[i], clusters[j], sim_matrix, static_embeds)
                cluster_dist[i, j] = csim
                cluster_dist[j, i] = csim
    return cluster_dist


def merge_cluster(cluster1: CloudCluster, cluster2: CloudCluster, cloud, static_embeds, keyword):
    k = cluster1.k
    ids = cluster1.points + cluster2.points
    intersection = set(cluster1.points) & set(cluster2.points)
    # print("intersection len = ", len(intersection))
    assert len(intersection) == 0
    assert cluster1.k == cluster2.k
    assert len(ids) == len(cluster1.points) + len(cluster2.points)
    points_embed = [cloud[i] for i in ids]
    cpoint = np.mean(points_embed, axis=0)
    nbs = clear_sim_words(keyword, cpoint, k, static_embeds)
    sim2c = [utils.cosine_similarity(cloud[i], cpoint) for i in ids]
    new_cluster = CloudCluster(cpoint, nbs, ids, k, sim2c)
    return new_cluster


def merge_cluster_once(clusters, cluster_simdist, cloud, static_embeds, keyword, drop=False):
    # print(" .. merge_cluster_once start")
    arg = np.argmax(cluster_simdist)
    # print("arg = ", arg)
    arg = np.unravel_index(arg, cluster_simdist.shape)
    # print("arg = ", arg)
    # print("merge max = ", cluster_simdist[arg])
    # # print("vsim for max = ", utils.cosine_similarity(cloud[arg[0]], cloud[arg[1]]))
    # print("nbs0 = ", clusters[arg[0]].c_neighbors)
    # print("nbs1 = ", clusters[arg[1]].c_neighbors)
    drop_st = []
    if not drop:
        m_cluster = merge_cluster(clusters[arg[0]], clusters[arg[1]],
                                  cloud, static_embeds, keyword)
    else:
        clu0 = clusters[arg[0]]
        clu1 = clusters[arg[1]]
        print("drop sim ", cluster_simdist[arg])
        if clu0.point_num > clu1.point_num and clu0.point_num/clu1.point_num > 2:
            print("drop clu1 ", clu1.c_neighbors, " whereas clu0 ", clu0.c_neighbors)
            m_cluster = clusters[arg[0]]
            drop_st += [clusters[arg[1]]]
        elif clu1.point_num > clu0.point_num and clu1.point_num/clu0.point_num > 2:
            print("drop clu0 ", clu0.c_neighbors, " whereas clu1 ", clu1.c_neighbors)
            m_cluster = clusters[arg[1]]
            drop_st += [clusters[arg[0]]]
        else:
            print("undrop, sim", cluster_simdist[arg],
                  " merge clu0 ", clu0.c_neighbors, " and clu1 ", clu1.c_neighbors)
            m_cluster = merge_cluster(clusters[arg[0]], clusters[arg[1]],
                                      cloud, static_embeds, keyword)

    # print(m_cluster.__dict__)
    # print("nbs new ", m_cluster.c_neighbors)
    new_clusters = []
    for i, clu in enumerate(clusters):
        if i == arg[0] or i == arg[1]:
            continue
        new_clusters.append(clu)
    new_clusters.append(m_cluster)
    # print("cluster_simdist shape ", cluster_simdist.shape)
    rip_dist = cluster_simdist
    rip_dist = np.delete(rip_dist, [arg[0], arg[1]], axis=0)
    rip_dist = np.delete(rip_dist, [arg[0], arg[1]], axis=1)
    # print("rip_dist shape ", rip_dist.shape)
    # print(" .. merge_cluster_once end")
    return new_clusters, rip_dist, drop_st


def cloud_clusters_reduce(clusters, cluster_simdist, sim_matrix, cloud, static_embeds, keyword, rt=0.9, drop=False):
    drop_semantic = []
    while np.max(cluster_simdist) > rt:
        # if len(clusters) % 20 == 0:
        #     print("cloud_clusters_reduce clusters num {} max sim {}".format(np.max(cluster_simdist), len(clusters)))
        clusters, rip_dist, drop_st =\
            merge_cluster_once(clusters, cluster_simdist, cloud, static_embeds, keyword, drop=drop)
        # if drop:
        #     ts.plt_weights(rip_dist, "rip_dist")
        cluster_simdist = clusters_distances(clusters, sim_matrix, rip_dist, static_embeds)
        drop_semantic += drop_st
        # if drop:
        #     ts.plt_weights(cluster_simdist, "cluster_simdist")
        # print("clusters len", len(clusters))
        # print(" new cluster simdist shape ", cluster_simdist.shape)
        # ts.plt_weights(new_cluster_simdist)
        # return clusters, cluster_simdist

    return clusters, cluster_simdist, drop_semantic


def subsmantic_from_clusters(clusters, cluster_simdist, sim_matrix,
                             cloud, static_embeds, keyword, sct=0.6):
    hf_clusters = []
    lf_semantic = []
    for i, clu in enumerate(clusters):
        # generaly hf_t is relate to the number of cloud points and the rt threshold
        # the higher points number the higher hf_t,
        # the lower rt threshold the higher hf_t.
        # it's hard to make a single best choice for different corpus.
        hf_t = len(cloud)//100 if len(cloud)//100 > 10 else 10
        if len(clu.points) >= hf_t:
            hf_clusters.append(clu)
            print(" Pending hf cid {} points {} ".format(i, clu.points))
            print(clu.c_neighbors)
            # print("sim ", clu.sim2c)
        elif len(clu.points) > 1:
            lf_semantic.append(clu)
    print("length lf_semantic before", len(lf_semantic))
    hf_cluster_simdist = clusters_distances(hf_clusters, sim_matrix, None, static_embeds)

    hf_semantic, polysemy_simdist, drop_semantic = \
        cloud_clusters_reduce(hf_clusters, hf_cluster_simdist, sim_matrix,
                              cloud, static_embeds, keyword, rt=sct, drop=True)
    ts.plt_weights(polysemy_simdist)
    lf_semantic += drop_semantic

    print("length lf_semantic after", len(lf_semantic))
    print("hf_semantic len", len(hf_semantic))
    for i, clu in enumerate(hf_semantic):
        print(">>>>> high cid {} points {} ".format(i, clu.points))
        print(clu.c_neighbors)

    for i, clu in enumerate(lf_semantic):
        print(">>>>> low cid {} points {} ".format(i, clu.points))
        print(clu.c_neighbors)

    # ts.plt_weights(polysemy_simdist)
    return hf_semantic, lf_semantic


def sub_semantic_similarity_matrix(subsm_set1, subsm_set2, static_embeds):
    sim_matrix = np.zeros((len(subsm_set1), len(subsm_set2)))
    for m, subsm1 in enumerate(subsm_set1):
        for n, subsm2 in enumerate(subsm_set2):
            sim = neighbor_similarity(subsm1, subsm2, static_embeds)
            sim_matrix[m, n] = sim
    return sim_matrix


def sc_from_similarity_matrix(hf_sims, hf1_lf2_sims, hf2_lf1_sims, rt=0.73, sct=0.6, use_lf=True):
    a = hf_sims.shape[0]
    b = hf_sims.shape[1]
    print(" sc_from_similarity_matrix rt {} , sct {},  use_lf {}".format(rt, sct, use_lf))
    for i in range(a):
        simmax = np.max(hf_sims[i, :])
        if simmax < sct:
            if use_lf:
                if hf1_lf2_sims.shape[1] == 0:
                    return 1
                lf_simmax = np.max(hf1_lf2_sims[i, :])
                if lf_simmax < rt:
                    print("sc change hf1_lf2_sims lf_simmax ", lf_simmax)
                    return 1
            else:
                print("sc change hf_sims in hf1 simmax ", simmax)
                return 1
    for i in range(b):
        simmax = np.max(hf_sims[:, i])
        if simmax < sct:
            if use_lf:
                if hf2_lf1_sims.shape[1] == 0:
                    return 1
                lf_simmax = np.max(hf2_lf1_sims[i, :])
                if lf_simmax < rt:
                    print("sc change hf2_lf1_sims lf_simmax ", lf_simmax)
                    return 1
            else:
                print("sc change hf_sims in hf2 simmax ", simmax)
                return 1

    return 0


def semantic_change_cloud_prior(hf_semantic1, lf_semantic1, hf_semantic2, lf_semantic2,
                                static_embeds, rt=0.73, sct=0.6):
    hf_sms1 = [hf.c_neighbors for hf in hf_semantic1]
    hf_sms2 = [hf.c_neighbors for hf in hf_semantic2]
    lf_sms1 = [hf.c_neighbors for hf in lf_semantic1]
    lf_sms2 = [hf.c_neighbors for hf in lf_semantic2]
    print(hf_sms1)
    print(hf_sms2)
    print(lf_sms1)
    print(lf_sms2)
    hf_sims = sub_semantic_similarity_matrix(hf_sms1, hf_sms2, static_embeds)
    hf1_lf2_sims = sub_semantic_similarity_matrix(hf_sms1, lf_sms2, static_embeds)
    hf2_lf1_sims = sub_semantic_similarity_matrix(hf_sms2, lf_sms1, static_embeds)
    print(hf_sims)
    ts.plt_weights(hf_sims)
    # ts.plt_weights(hf1_lf2_sims)
    # ts.plt_weights(hf2_lf1_sims)
    semantic_change = \
        sc_from_similarity_matrix(hf_sims, hf1_lf2_sims, hf2_lf1_sims, rt=rt, sct=sct, use_lf=True)

    grade_nb, grade_emb, grade_nb_hf = st2.sc_grade(hf_semantic1, hf_semantic2, hf_sims, static_embeds)
    return semantic_change, grade_nb_hf


def cloud_cluster_rt(sim_matrix, nbs_cloud, cloud, static_embeds, keyword, rt, k, time, language):
    rt_clusters_pth = './data/{}/words/{}/cloud_sim_matrix/rt{}/k{}_rt_clusters_t{}.pkl'. \
        format(language, keyword, rt, k, time)
    rt_cluster_simdist_pth = './data/{}/words/{}/cloud_sim_matrix/rt{}/k{}_rt_cluster_simdist_t{}.pkl'. \
        format(language, keyword, rt, k, time)
    clusters = utils.load_from_disk(rt_clusters_pth)
    cluster_simdist = utils.load_from_disk(rt_cluster_simdist_pth)
    if clusters is not None and cluster_simdist is not None:
        print(" Load cached rt clusters for word ".format(keyword))
        return clusters, cluster_simdist

    clusters = []
    for i, point in enumerate(cloud):
        clusters.append(CloudCluster(point, nbs_cloud[i], [i], k, [1]))

    # test cluster_simdist is same with sim_matrix
    # cluster_simdist = clusters_distances(clusters, sim_matrix)
    # print(" cluster_simdist shape ", cluster_simdist.shape)
    # ts.plt_weights(sim_matrix - cluster_simdist)
    cluster_simdist = sim_matrix.copy()
    row, col = np.diag_indices_from(cluster_simdist)
    cluster_simdist[row, col] = 0
    # ts.plt_weights(sim_matrix - cluster_simdist)

    clusters, cluster_simdist, _ = \
        cloud_clusters_reduce(clusters, cluster_simdist, sim_matrix,
                              cloud, static_embeds, keyword, rt=rt)

    utils.save_to_disk(rt_clusters_pth, clusters)
    utils.save_to_disk(rt_cluster_simdist_pth, cluster_simdist)
    return clusters, cluster_simdist


def cloud_cluster(cloud, static_embeds, keyword, language, time, k,
                  usecache=False, subsample=True, rt=0.73, sct=0.6):
    hf_path = './data/{}/words/{}/cloud_sim_matrix/sct{}/rt{}/k{}_cloud_hfsc_t{}.pkl'.\
        format(language, keyword, sct, rt, k, time)
    lf_path = './data/{}/words/{}/cloud_sim_matrix/sct{}/rt{}/k{}_cloud_lfsc_t{}.pkl'.\
        format(language, keyword, sct, rt, k, time)

    hf_semantic = utils.load_from_disk(hf_path)
    lf_semantic = utils.load_from_disk(lf_path)
    if usecache and hf_semantic is not None and lf_semantic is not None:
        print(" >> Load hf_semantic and lf_semantic from cache for word '{}' ".format(keyword))
        return hf_semantic, lf_semantic

    sim_matrix, nbs_cloud = \
        gen_cloud_distance(cloud, static_embeds, keyword, language, time=time, k=k)
    # ts.plt_weights(sim_matrix)
    # only_cloud_distance = True
    # if only_cloud_distance:
    #     return

    if subsample and len(nbs_cloud) > 1200:
        sim_matrix = sim_matrix[:1200, :1200]
        nbs_cloud = nbs_cloud[:1200]
        cloud = cloud[:1200]

    clusters, cluster_simdist =\
        cloud_cluster_rt(sim_matrix, nbs_cloud, cloud, static_embeds, keyword, rt, k, time, language)
    print("Reduce clusters num: ", len(clusters))

    hf_semantic, lf_semantic = \
        subsmantic_from_clusters(clusters, cluster_simdist, sim_matrix,
                                 cloud, static_embeds, keyword, sct=sct)
    # ts.plt_weights(cluster_simdist)
    utils.save_to_disk(hf_path, hf_semantic)
    utils.save_to_disk(lf_path, lf_semantic)
    return hf_semantic, lf_semantic


if __name__ == '__main__':
    semeval_cloud_distance()
    # evaluate_task1()
