from transformers import pipeline

clf = pipeline(
    task = 'sentiment-analysis', 
    model = 'SkolkovoInstitute/russian_toxicity_classifier')

text = ['Ты идиот, если так считаешь!',
    	'Я вас люблю']

clf(text)

print(clf(text))