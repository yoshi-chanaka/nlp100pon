import CaboCha
from graphviz import Digraph
from nlp100pon_41 import extractParsedInfo
from nlp100pon_42 import organizeSentenceInfo


def sentence2graph(input_sentence, output_path='../../materials/chap05s2g_tmp.txt'):
    # 4.7 コードを段落に分ける

    # Cabochaによる解析
    paser = CaboCha.Parser()
    tree = paser.parse(input_sentence)
    result_parsing = tree.toString(CaboCha.FORMAT_LATTICE)

    # 解析結果を一度ファイルに保存
    with open(output_path, 'w', encoding='utf-8') as fo:
        for line in result_parsing.split('\n'):
            fo.write(line + '\n')

    # 以下，問42と同じ流れ
    # 文章ごとにextractParsedInfoで解析結果を分け，
    # 各々からextractPhrasesDependenciesで文節と文節間の係り受け関係を抽出
    phrases_list_all, dependencies_list_all = [], []
    sentences_info = extractParsedInfo(path=output_path)

    for info_in_sentence in sentences_info:
        organized_info_in_sentence = organizeSentenceInfo(info_in_sentence)

        phrases = [''.join(phr_list) for phr_list in organized_info_in_sentence['surface']]
        phrases_list_all.append(phrases)
        dependencies_list_all.append(organized_info_in_sentence['dependencies'])

    return phrases_list_all, dependencies_list_all

# 10章　無関係の下位問題を抽出する．
# ノードのラベルとエッジから有向グラフを作成する関数
def visualizeTree(edges, node_labels, output='digraph'):

    """
    node: ['太郎は', '花子が', '読んでいる', '本を', '次郎に', '渡した']
    edges: [[0, 5], [1, 2], [2, 3], [3, 5], [4, 5]]
    """

    dg = Digraph(format='png')
    dg.attr('node', fontname="MS Gothic") # ノードのフォン地をMSゴシックに変換
    num_nodes = len(node_labels)

    for idx_node in range(num_nodes):
        dg.node(str(idx_node), label=node_labels[idx_node])

    for idx_src, idx_dst in edges:
        dg.edge(str(idx_src), str(idx_dst))

    dg.render(output)


if __name__ == "__main__":

    input_sentence = \
        '太郎は花子が読んでいる本を次郎に渡した。'
    phrases_all, dependence_all = sentence2graph(input_sentence)
    for phrases, dependencies in zip(phrases_all, dependence_all):
        visualizeTree(
            edges=dependencies,
            node_labels=phrases,
            output='../../figures/chap05graph_book'
        )

    input_sentence = \
        'ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。'
    phrases_all, dependence_all = sentence2graph(input_sentence)
    for phrases, dependencies in zip(phrases_all, dependence_all):
        visualizeTree(
            edges=dependencies,
            node_labels=phrases,
            output='../../figures/chap05graph_ai'
        )