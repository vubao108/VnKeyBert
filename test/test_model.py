import sys, os
# root_dir = os.path.dirname(os.path.dirname(__file__))
# vnkeybert_folder = os.path.join(root_dir,'vnkeybert') 
# sys.path.insert(0,vnkeybert_folder)

from  vnkeybert import VnKeyBERT

if __name__ == '__main__':
    docs = ['đầm thiết kế giấu vết xước tốt, đuôi cá nhẹ nhàng giúp nổi bật. người mẫu trong bộ váy này khiến cô gái trở nên dễ thương một cách kỳ lạ.']
    
    vnKeyBERT = VnKeyBERT()
    while True:
        doc = input('nhập văn bản < 255 ky tự: ').strip()
        keys = vnKeyBERT.extract_keywords(docs=[doc], keyphrase_ngram_range=(1,4),use_maxsum=True, top_n=10)
        print(keys)
