VnKeyBert use phoBert to extract Vietnamese keywords.
### install:
pip install git+https://github.com/vubao108/VnKeyBert.git

### use:


```python
from  vnkeybert import VnKeyBERT

vnKeyBERT = VnKeyBERT()
docs = ['đầm thiết kế giấu vết xước tốt, đuôi cá nhẹ nhàng giúp nổi bật.']
keys = vnKeyBERT.extract_keywords(docs=[text], keyphrase_ngram_range=(1,4),use_maxsum=True, top_n=5)
print(keys)
```
