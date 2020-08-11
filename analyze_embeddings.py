from gensim.models import Word2Vec
import pandas as pd
import csv


model = Word2Vec.load('./Word_Embeddings_Model/model.bin')
print(model)
words = list(model.wv.vocab)
print(words)
past_tsv_file=pd.read_csv('./training_files/new_train_file_tok.tsv',delimiter='\t',encoding='utf-8')
new_tsv_file=open("./training_files/embedding_train_file_tok.tsv",'w')

appended=[]


for row in past_tsv_file.itertuples():
    # print(row)
    new_tsv_file.write(row[1]+"\t"+row[2]+"\n")
    appended.append([row[1].strip(),row[2].strip()])
new_tsv_file.close()


for row in past_tsv_file.itertuples():
    if(row[2].strip() != 'O'):
        if(row[1].strip() in words):
            most_similar_words = model.most_similar(row[1].strip(), topn=2)
            for word,similarity in most_similar_words:
                if([word.strip(),row[2].strip()] not in appended):
                    new_tsv_file=open("./training_files/embedding_train_file_tok.tsv",'a')
                    new_tsv_file.write(word.strip()+"\t"+row[2].strip()+"\n")
                    appended.append([word,row[2].strip()])
    # new_tsv_file.write(row[1]+"\t"+row[2]+"\n")
new_tsv_file.close()





# most_similar_words = model.most_similar( 'staff', topn=1)

# print(most_similar_words)
