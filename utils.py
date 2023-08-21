import pickle
import os
import numpy as np
from scipy.optimize import linear_sum_assignment


def all_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def find_files(dir):
    files = []
    for file in all_files(dir):
        # convert to linux file path
        file = '/'.join(file.split('\\'))
        # print(file)
        files.append(file)
    return files


def stop_words(language=None):
    support_languages = {'en': "./corpus/stop_words_english-small.txt"}
    if language not in support_languages:
        print("Unsupported stop word for language ", language)
        return []
    with open(support_languages['en'], "r", encoding="utf-8") as f:
        swl = f.read().split('\n')
    return swl


def rm_stop_words(sentence, stop_words_list):
    new_sentence = [word for word in sentence if word not in stop_words_list]
    return new_sentence


def remove_stop_words(corpus, language):
    stop_words_list = stop_words(language)
    new_corpus = []
    for sentence in corpus:
        new_corpus.append(rm_stop_words(sentence, stop_words_list))
    return new_corpus


def prepare_coupus_from_txt(data_file, language):
    with open(data_file, "r", encoding="utf-8") as corpus_data:
        corpus_raw = corpus_data.read().split('\n')
    corpus_cut = [sentence.split() for sentence in corpus_raw]
    corpus = remove_stop_words(corpus_cut, language)
    return corpus


def create_filepath_dir(filepath):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)


def exists(f):
    return os.path.exists(f)


def save_to_disk(pickle_f, obj):
    dir = os.path.dirname(pickle_f)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(pickle_f, "wb") as f:
        pickle.dump(obj, f)


def save_as_txt(path, obj):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path, "w") as f:
        f.write(str(obj))


def save_sendata_as_txt(path, sendata):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    data_str = ''
    for sen in sendata:
        data_str += sen
        data_str += '\n'
    with open(path, "w", encoding='utf-8') as f:
        f.write(data_str)


def load_from_disk(pickle_f):
    if not os.path.exists(pickle_f):
        print("pickle_f not exists ", pickle_f)
        return None
    with open(pickle_f, "rb") as f:
        obj = pickle.load(f)
    return obj


def picklebig(obj, file):
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.makedirs(dir)

    max_bytes = 2 ** 31 - 1
    # write
    bytes_out = pickle.dumps(obj)
    with open(file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def unpicklebig(file):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def word_voc(word, voc_index, embeddings):
    id = voc_index[word]
    return embeddings[id]


def cosine_similarity(v_w1, v_w2):
    theta_sum = np.dot(v_w1, v_w2)
    theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
    theta = theta_sum / theta_den
    return theta


def word_cosine_similarity(w1, w2, voc_index, embeddings):
    v_w1 = word_voc(w1, voc_index, embeddings)
    v_w2 = word_voc(w2, voc_index, embeddings)
    theta = cosine_similarity(v_w1, v_w2)
    return theta


def most_sim_words(v_w1, top_n, voc_index, embeddings, willprint=True):
    word_sim = {}
    for word_c in voc_index:
        v_w2 = word_voc(word_c, voc_index, embeddings)
        theta = cosine_similarity(v_w1, v_w2)
        word_sim[word_c] = theta
    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
    words = []
    for word, sim in words_sorted[:top_n]:
        if willprint:
            print(word, sim)
        words.append((word, sim, word_voc(word, voc_index, embeddings)))

    return words


def word_sim(word, top_n, voc_index, embeddings):
    print("+++++++++++ eva word_sim ", word)
    v_w1 = word_voc(word, voc_index, embeddings)
    sim_words = most_sim_words(v_w1, top_n, voc_index, embeddings)
    return sim_words


def test_embeds_sim(key_word):
    index_voc, voc_index, embeddings = load_from_disk("de_static_embed.pkl")
    word_sim(key_word, 20, voc_index, embeddings)
    target_embeds = load_from_disk("Maus_de_wikiembeds.pkl")
    for i, embed in enumerate(target_embeds):
        print("++++++++ ++++++ embed i ", i)
        most_sim_words(embed, 3, voc_index, embeddings)


def words_sim(w1,w2,voc_index, embeddings):
    wv1 = word_voc(w1, voc_index, embeddings)
    wv2 = word_voc(w2, voc_index, embeddings)
    theta_sum = np.dot(wv1, wv2)
    theta_den = np.linalg.norm(wv1) * np.linalg.norm(wv2)
    theta = theta_sum / theta_den
    return theta


def read_coupus_from_txt(key_word, data_file):
    with open(data_file, "r", encoding="utf-8") as corpus_data:
        corpus_raw = corpus_data.read().split('\n')
    data_list = []
    for sentence in corpus_raw:
        if key_word in sentence:
            data_list.append(sentence)
    print("data_list len ", len(data_list))
    return data_list


def init_static_embedding():
    index_voc = {}
    voc_index = {}
    embeddings = {}
    static_embeds = (index_voc, voc_index, embeddings)
    return static_embeds


def merge_static_embeds(static_embeds1, static_embeds2):
    static_embeds = init_static_embedding()
    index_voc, voc_index, embeddings = static_embeds
    index_voc1, voc_index1, embeddings1 = static_embeds1
    index_voc2, voc_index2, embeddings2 = static_embeds2
    voc_len = 0
    for word in voc_index1:
        if word not in voc_index:
            voc_index[word] = voc_len
            index_voc[voc_len] = word
            embeddings[voc_len] = embeddings1[voc_index1[word]]
            voc_len += 1

    for word in voc_index2:
        if word not in voc_index:
            voc_index[word] = voc_len
            index_voc[voc_len] = word
            embeddings[voc_len] = embeddings2[voc_index2[word]]
            voc_len += 1

    return static_embeds


def has_num(w):
    number_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for num in number_tokens:
        if num in w:
            return True
    return False


def eva_uncommon_word(w, keyword, l_min=2, language="en"):
    special_tokens = [keyword, "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    stop_words_set = stop_words(language)
    if w in special_tokens:
        return True
    if w in stop_words_set:
        return True
    if len(w) < l_min:
        return True
    if has_num(w[0]):
        return True


def clear_sim_words(keyword, embed, simsize, static_embeds):
    index_voc, voc_index, embeddings = static_embeds
    sizegap = 20
    words = most_sim_words(embed, simsize + sizegap, voc_index, embeddings, willprint=False)
    poly_sims = []
    for i, w in enumerate(words):
        # print('id: {} w: {} sim: {}'.format(i, w[0], w[1]))
        if eva_uncommon_word(w[0], keyword, l_min=2, language="en"):
            continue
        if len(poly_sims) < simsize:
            poly_sims.append(w[0])

    return poly_sims


def eva_polysim_linear_assignment(polyset1, polyset2, static_embeds):
    index_voc, voc_index, embeddings = static_embeds
    weights = np.zeros((len(polyset1), len(polyset2)))
    for i, w_row in enumerate(polyset1):
        for j, w_col in enumerate(polyset2):
            weight = word_cosine_similarity(w_row, w_col, voc_index, embeddings)
            weights[i, j] = weight
    row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
    mean = weights[row_ind, col_ind].mean()
    # sum = weights[row_ind, col_ind].sum()
    # print("keyword weights : ", weights)
    # print("keyword weights sum {} mean {}", sum, mean)
    # print("linear_assignment match weights max {} min {}".
    #       format(weights[row_ind, col_ind].max(), weights[row_ind, col_ind].min()))

    return mean


def check_gpu():
    import torch
    print(torch.cuda.is_available())  # true 查看GPU是否可用

    print(torch.cuda.device_count())  # GPU数量， 1

    print(torch.cuda.current_device())  # 当前GPU的索引， 0

    print(torch.cuda.get_device_name(0))  # 输出GPU名称


# test_embeds_sim('mouse')
# test_embeds_sim('Maus')

# index_voc, voc_index, embeddings = load_from_disk("de_static_embed.pkl")
# print('words_sim', words_sim('Maus','Tastatur',voc_index, embeddings))

