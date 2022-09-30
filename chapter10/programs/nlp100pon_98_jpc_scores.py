import matplotlib.pyplot as plt

"""
JParaCrawlのスコアをヒストグラムで表示
"""

if __name__ == "__main__":

    filepath = '../data/jpc-data/en-ja/en-ja.bicleaner05.txt'
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            info = line.strip().split('\t')
            scores.append(float(info[2]))
            line = f.readline()
    print(min(scores), max(scores))
    plt.hist(scores, bins=30)
    plt.savefig('../figures/jpc_scores.png')
    # 0.5 0.797
