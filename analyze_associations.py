import pandas as pd
import csv

def main():
    past_tsv_file=pd.read_csv('./training_files/train_file_tok.tsv',delimiter='\t',encoding='utf-8')
    new_tsv_file=open("./training_files/new_train_file_tok.tsv",'w')
    #copying contents of old training file into new training file
    copy_contents(past_tsv_file,new_tsv_file)
    new_tsv_file.close()
    #opening associations file
    ifile = open('./training_files/associations.out', 'r', encoding = "ISO-8859-1").read()
    #adding more training data from associations
    analyze_associations(ifile,past_tsv_file,new_tsv_file)

def analyze_associations(ifile, past_tsv_file, new_tsv_file):
    ifile_lines=ifile.splitlines()
    for iline in ifile_lines:
        new_tsv_file = open("./training_files/new_train_file_tok.tsv",'a')
        #splitting first word
        iarray=iline.split('(')[0].strip().split()
        flag_token1=0
        flag_token2=0
        for row1 in past_tsv_file.itertuples():
            #if first word is present in the training file
            if((iarray[0].strip().lower() == row1[1].strip().lower())):
                #and if first word's tagged Named Entity Class is 'O'
                if(row1[2].strip() != 'O'):
                    flag_token1=3
                    token1_tag = row1[2].strip()
                    break
                else:
                    flag_token1=2
                    token1_tag = row1[2].strip()
            else:
                flag_token1=1
        for row2 in past_tsv_file.itertuples():
            #if second word is present in the training file
            if((iarray[1].strip().lower() == row2[1].strip().lower())):
                #and if second word's tagged Named Entity Class is 'O'
                if(row2[2].strip() != 'O'):
                    flag_token2=3
                    token2_tag = row2[2].strip()
                    break
                else:
                    flag_token2=2
                    token2_tag = row2[2].strip()
            else:
                flag_token2=1

        #if first word is in the training file and second word has no specific Named Entity Class tagged to it
        if(flag_token1 == 3 and (flag_token2 == 1 or flag_token2 == 2 or (flag_token2 == 3 and token1_tag != token2_tag))):
            flag_same_tag=0
            append_tsv_file=pd.read_csv('./training_files/train_file_tok.tsv',delimiter='\t',encoding='utf-8')
            for row in append_tsv_file.itertuples():
                if(iarray[1].strip().lower() == row[1].strip().lower() and token2_tag == row[2].strip().lower()):
                    flag_same_tag=1
            if(flag_same_tag == 0 and token2_tag != 'O'):
                new_tsv_file.write(iarray[1].strip().lower()+"\t"+token2_tag+"\n")

        #if first word is in the training file and second word has no specific Named Entity Class tagged to it
        if(flag_token2 == 3 and (flag_token1 == 1 or flag_token1 == 2 or (flag_token1 == 3 and token1_tag != token2_tag))):
            flag_same_tag=0
            append_tsv_file=pd.read_csv('./training_files/train_file_tok.tsv',delimiter='\t',encoding='utf-8')
            for row in append_tsv_file.itertuples():
                if(iarray[0].strip().lower() == row[1].strip().lower() and token1_tag == row[2].strip().lower()):
                    flag_same_tag=1
            if(flag_same_tag == 0 and token1_tag != 'O'):
                new_tsv_file.write(iarray[0].strip().lower()+"\t"+token1_tag+"\n")
        new_tsv_file.close()

def copy_contents(past_tsv_file,new_tsv_file):
    for row in past_tsv_file.itertuples():
        new_tsv_file.write(row[1]+"\t"+row[2]+"\n")

if __name__ == "__main__":
    main()
