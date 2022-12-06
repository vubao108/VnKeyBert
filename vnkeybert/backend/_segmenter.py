import os
import py_vncorenlp
from typing import List


class Segmenter:
  def __init__(self) :
    parent_dir = os.path.dirname(__file__)
    vncorelp_dir = os.path.join(parent_dir,'vncorenlp')
    py_vncorenlp.download_model(save_dir=vncorelp_dir)
    self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorelp_dir)
  
  def get_segment(self, documents:List[str])->List[str]:
    documents_segmented = [] 
    for doc in documents:
      segments = self.rdrsegmenter.word_segment(doc)
      doc_segmented = ' '.join(segments)
      documents_segmented.append(doc_segmented)
    
    return documents_segmented


if  __name__ == '__main__':
  ## test
  segmenter = Segmenter()
  doc = 'đầm thiết kế giấu vết xước tốt, đuôi cá nhẹ nhàng giúp nổi bật. người mẫu trong bộ váy này khiến cô gái trở nên dễ thương một cách kỳ lạ.'
  docs = segmenter.get_segment([doc])
  print(docs)
