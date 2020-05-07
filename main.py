from flask import Flask, render_template, request
from deeppavlov import build_model, configs

app = Flask(__name__)

model_qa = build_model(configs.squad.squad_bert, download=True)

with open('corpus.txt', 'r') as fp:
    corpus = fp.readlines()
    corpus = "".join(corpus)

@app.route('/')
def index():
    context = {}
    if request.args.get('question'):
        question = request.args.get('question')
        context['question'] = question
        context['answer'] = model_qa([corpus], [question])[0][0]
    return render_template('index.html', context=context)

if __name__ == '__main__':
    app.run(host='localhost')