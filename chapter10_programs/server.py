from load import load_data
from translate import load_model, translate

from flask import Flask, render_template, request
import sentencepiece as spm
import torch

app = Flask(__name__)

@app.route("/", methods=["GET"])
def get():
    return render_template("index.html", \
        message = "テキストを入力してください")

@app.route("/", methods=["POST"])
def post():
    text = request.form["name"]
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(text))
    output = translate(
        model=model, src_sentence=proc_text, 
        vocab_src=vocab_ja, vocab_tgt=vocab_en, 
        device=device, post_proc=True
    )
    return render_template(
        "index.html", \
        input = f"入力:\t{text}", \
        output = f"出力:\t{output}", \
        message = "テキストを入力してください"
    )
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))
    
    sp_tokenizer = spm.SentencePieceProcessor(model_file='../models/kftt_sp_ja.model')

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    app.run()
