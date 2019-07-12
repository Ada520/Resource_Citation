
import sys
import os
import codecs
import random
# create a mini dataset for test code
PATH="F:/Github/Resource_Citation/data/SciRes"
def get_file():
    files=os.listdir(PATH)
    return files
files=get_file()
for path in files:
    path=PATH+"/"+path
    root, extension = os.path.splitext(path)
    minidata_path=root+"2"+extension
    outf=open(minidata_path,'w',encoding='utf-8')
    print(path)
    with open(path,encoding='utf-8') as f:
        lines=f.readlines()
        random.shuffle(lines)
        for line in lines[:100]:
            outf.write(line)
    outf.close()



