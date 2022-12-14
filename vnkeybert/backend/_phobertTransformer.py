
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

        


    def embed(self, documents: List[str], max_len = 150, is_verborse = False) -> np.ndarray:
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

        max_token_len = 0
        for doc in documents:
            doc_id = self.tokenizer.encode(doc)
            documents_ids.append(doc_id)
            if max_token_len < len(doc_id):
                max_token_len = len(doc_id)
            
        max_len = max_len if max_len < max_token_len else max_token_len
        documents_ids = pad_sequences(documents_ids, maxlen = max_len, dtype="long", value = 0, truncating = "post", padding = "post")
        print(documents_ids.shape)

        if is_verborse:
            for item_ids in documents_ids:
              print(self.tokenizer.decode(item_ids))

        docs_mask = []
        for item_ids in documents_ids:
            mask = [int(token_id>0) for token_id in item_ids]
            docs_mask.append(mask)
        docs_mask =  torch.tensor(np.array(docs_mask))
        docs_input = torch.tensor(documents_ids)

        embeddings = self.embedding_model(input_ids = docs_input, attention_mask= docs_mask)
        
        return embeddings.last_hidden_state
  
    def embed_cls(self, documents: List[str], max_len = 150, is_verbose = False):
        if max_len > 255 and max_len < 1:
             raise ValueError(
                "max_len must < 255 "   
            )
        embedding = self.embed(documents, max_len=max_len, is_verborse= is_verbose)
        return embedding[:,0,:]
    
    def get_segment(self, docs:List[str]) -> List[str]:
        return self.segmenter.get_segment(docs)

if __name__ == '__main__':
  phobert = PhoBertTransformerBackend()
  documents = ['H??m_nay tr???i n??ng qu?? n??n t??i ??? nh?? !',
  'ng?????i_m???u xinh trong b??? v??y n??y khi???n c?? g??i tr???_n??n d???_th????ng m???t_c??ch k???_l???  .'
  ]
  phobert.embed_cls(documents, is_verbose=True)


  
  



