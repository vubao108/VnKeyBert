import sys, os
# root_dir = os.path.dirname(os.path.dirname(__file__))
# vnkeybert_folder = os.path.join(root_dir,'vnkeybert') 
# sys.path.insert(0,vnkeybert_folder)

from  vnkeybert import VnKeyBERT


import json
import random
import os
current_dir = os.path.dirname(__file__)
data_file = os.path.join(current_dir,'test_data.json' )

def get_random_text():
  with open(data_file, encoding='utf8') as f:
    data = json.load(f)
    key, list_sentence = random.choice(list(data.items()))
  
  return key, random.choice(list_sentence)



if __name__ == '__main__':
    docs = ['đầm thiết kế giấu vết xước tốt, đuôi cá nhẹ nhàng giúp nổi bật. người mẫu trong bộ váy này khiến cô gái trở nên dễ thương một cách kỳ lạ.']
    
    vnKeyBERT = VnKeyBERT()
    while True:
        input("Enter de tiep tuc")
        key, text = get_random_text()
        print('key: '+ key)
        print("text: " +text)
        keys = vnKeyBERT.extract_keywords(docs=[text], keyphrase_ngram_range=(1,4),use_maxsum=True, top_n=5)
        print(keys)
