import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from typing import List, Union, Tuple

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from vnkeybert._mmr import mmr
from vnkeybert._maxsum import max_sum_distance
from vnkeybert.backend._phobertTransformer import PhoBertTransformerBackend


class VnKeyBERT:
    """
    A minimal method for keyword extraction with phoBERT

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    First, document embeddings are extracted with phoBERT to get a
    document-level representation. Then, word embeddings are extracted
    for N-gram words/phrases. Finally, we use cosine similarity to find the
    words/phrases that are the most similar to the document.

    The most similar words could then be identified as the words that
    best describe the entire document.

    <div class="excalidraw">
    --8<-- "docs/images/pipeline.svg"
    </div>
    """

    def __init__(self, model="vinai/phobert-base"):
        self.model = PhoBertTransformerBackend(model)

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                  
            stop_words: Stopwords to remove from the document.
                        
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        from vnkeybert import VnKeyBERT
        doc = 'Trời hôm nay thật đẹp'
        kw_model = VnKeyBERT()
        keywords = kw_model.extract_keywords(doc)
        ```

        To extract keywords from multiple documents, which is typically quite a bit faster:

        ```python
        docs = ['Trời hôm nay thật đẹp', 'Chúc ngày mới tốt lành']
        from vnkeybert import VnKeyBERT

        kw_model = VnKeyBERT()
        keywords = kw_model.extract_keywords(docs)
        ```
        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        # Extract potential words using a vectorizer / tokenizer
        docs = self.model.get_segment(docs)   
        try:
            count = CountVectorizer(
                ngram_range=keyphrase_ngram_range,
                stop_words=None,
                min_df=min_df
            ).fit(docs)
        except ValueError:
            return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(docs)

        # Extract embeddings
        
        doc_embeddings, word_embeddings = self.extract_embeddings(docs,keyphrase_ngram_range= keyphrase_ngram_range, is_segmented = True)
        ## convert to numpy array 
        doc_embeddings = doc_embeddings.detach().numpy()
        word_embeddings = word_embeddings.detach().numpy()
        # Find keywords
        all_keywords = []
        for index, _ in enumerate(docs):

            try:
                # Select embeddings
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

                # Maximal Marginal Relevance (MMR)
                if use_mmr:
                    keywords = mmr(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        diversity,
                    )

                # Max Sum Distance
                elif use_maxsum:
                    keywords = max_sum_distance(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        nr_candidates,
                    )

                # Cosine-based keyword extraction
                else:
                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    keywords = [
                        (candidates[index], round(float(distances[0][index]), 4))
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]

                all_keywords.append(keywords)

            # Capturing empty keywords
            except ValueError:
                all_keywords.append([])

        # Highlight keywords in the document
        if len(all_keywords) == 1:
            all_keywords = all_keywords[0]

        return all_keywords

    def extract_embeddings(
        self,
        docs: Union[str, List[str]],
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        is_segmented: bool = False
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        if not is_segmented:
            docs = self.model.get_segment(docs)   
            
        try:
            count = CountVectorizer(
                ngram_range=keyphrase_ngram_range
            ).fit(docs)
        except ValueError:
            return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()

        doc_embeddings_cls = self.model.embed_cls(docs)
        word_embeddings_cls = self.model.embed_cls(words)

        return doc_embeddings_cls, word_embeddings_cls



if __name__ == '__main__':
    docs = ['đầm thiết kế giấu vết xước tốt, đuôi cá nhẹ nhàng giúp nổi bật. người mẫu trong bộ váy này khiến cô gái trở nên dễ thương một cách kỳ lạ.']
    
    vnKeyBERT = VnKeyBERT()
    while True:
        doc = input('nhập văn bản < 255 ky tự: ').strip()
        keys = vnKeyBERT.extract_keywords(docs=[doc], keyphrase_ngram_range=(1,4),use_maxsum=True, top_n=5)
        print(keys)
