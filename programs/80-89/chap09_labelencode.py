def MakeLabelEncoder(input_corpus):
    """
    input: ['今日', 'は', '良い', '天気', 'です', 'ね']
    """
    import collections
    import pandas as pd

    word_count_dict = dict(collections.Counter(input_corpus))
    df_word_count = pd.DataFrame({
        'word'  : list(word_count_dict.keys()),
        'count' : list(word_count_dict.values())
    })
    sorted_df_word_count = df_word_count.sort_values('count', ascending=False).reset_index(drop=True)

    label_encoder_dict, label_decoder_list = {}, ['[UNK]']

    # 出現頻度が2回未満の単語にID:0を付与
    for word, count in sorted_df_word_count[sorted_df_word_count['count'] < 2].values:
        label_encoder_dict[word] = 0
    
    # 出現頻度が2回以上の単語に対して出現頻度順にIDを付与
    for idx, (word, count) in enumerate(sorted_df_word_count[sorted_df_word_count['count'] >= 2].values):
        label_encoder_dict[word] = idx + 1
        label_decoder_list.append(word)
    
    return label_encoder_dict, label_decoder_list


if __name__ == "__main__":

    text = ['i', 'have', 'a', 'pen', 'i', 'have', 'an', 'apple', 'i', 'like', 'both']
    print(MakeLabelEncoder(text))

    # ({'a': 0, 'pen': 0, 'an': 0, 'apple': 0, 'like': 0, 'both': 0, 'i': 1, 'have': 2}, ['[UNK]', 'i', 'have'])
