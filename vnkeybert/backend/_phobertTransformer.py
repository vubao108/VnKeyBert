
import numpy as np
import os
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
from vnkeybert.backend._segmenter import Segmenter
from vnkeybert.backend._normaltext import NormalText
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import re

parent_dir = os.path.basename(os.path.dirname(__file__))
vncorelp_dir = os.path.join(parent_dir,'vncorenlp')

class PhoBertTransformerBackend():
    def __init__(self, embedding_model:str = 'vinai/phobert-base'):
        if embedding_model in ["vinai/phobert-base", "vinai/phobert-large"]:
            self.embedding_model = AutoModel.from_pretrained(embedding_model)
        else:
            raise ValueError(
                "Please select a correct phobert model" 
            
            )
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.segmenter = Segmenter()
        self.normalText = NormalText()

        


    def embed(self, documents: List[str], max_len = 150) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings
        Arguments:
            documents: A list of documents or words to be embedded
        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        
        documents = self.normalText.normal_text(documents)
        documents_ids = []
        for doc in documents:
            doc_id = self.tokenizer.encode(doc)
            documents_ids.append(doc_id)

        documents_ids = pad_sequences(documents_ids, maxlen = max_len, dtype="long", value = 0, truncating = "post", padding = "post")
        print(documents_ids.shape)

        docs_mask = []
        for item_ids in documents_ids:
            mask = [int(token_id>0) for token_id in item_ids]
            docs_mask.append(mask)
        docs_mask =  torch.tensor(np.array(docs_mask))
        docs_input = torch.tensor(documents_ids)

        embeddings = self.embedding_model(input_ids = docs_input, attention_mask= docs_mask)
        
        return embeddings.last_hidden_state
  
    def embed_cls(self, documents: List[str], max_len = 150):
        if max_len > 255 and max_len < 1:
             raise ValueError(
                "max_len must < 255 "   
            )
        embedding = self.embed(documents, max_len=max_len)
        return embedding[:,0,:]
    
    def get_segment(self, docs:List[str]) -> List[str]:
        return self.segmenter.get_segment(docs)

if __name__ == '__main__':
  phobert = PhoBertTransformerBackend()
  documents = ['Hôm_nay trời nóng quá nên tôi ở nhà viết Viblo!',
  'người_mẫu xinh trong bộ váy này khiến cô gái trở_nên dễ_thương một_cách kỳ_lạ  .'
  ]
  phobert.embed_cls(documents)


  
  



