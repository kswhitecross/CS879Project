"""
This file manages the creation of data for this project.  If run `python msmarco.py`, then it will create the data file
 for this project containing all of the queries, qrels, and documents necessary.
"""
import ir_datasets
import json
import os
import random
import bm25s
import Stemmer
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from typing import Union


class Stopper:
    def __init__(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Missing stopwords... Downloading")
            nltk.download('stopwords')
        self.stopwords = set(stopwords.words('english'))

    def stop(self, words: Union[str, list[str]]) -> list[str]:
        """
        Remove stopwords from words.  If words is a string, it will be word tokenized.  If words is a list of strings,
         then it will be treated as a list of tokens.
        """
        if isinstance(words, str):
            words = word_tokenize(words)
        return [w for w in words if not w.lower() in self.stopwords]


def main():
    # The meaning of life, the universe and everything... chr(42), aka '*'
    random.seed(42)
    np.random.seed(42)

    # download the nltk punkt tokenizer if we need to
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Missing tokenizer... Downloading")
        nltk.download('punkt')

    # load the selected queries
    print("Loading queries...")
    with open(os.path.join('data', 'selected_queries.json')) as f:
        # queries maps q_ids -> text
        queries = json.loads(f.read())

    # load the dataset
    dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')

    # build qrels dicts
    # bin_qrels is 0 or 1 relevance (used for sampling negatives)
    print("Building qrels...")
    scores = [0, 1, 2, 3]
    qrels = {q_id: {score: list() for score in scores} for q_id in queries.keys()}
    nonzero_qrels = {q_id: list() for q_id in queries.keys()}
    for q_id, d_id, rel, _ in dataset.qrels_iter():
        if q_id in qrels.keys():
            qrels[q_id][rel].append(d_id)
            if rel > 0:
                nonzero_qrels[q_id].append(d_id)

    # sample qrels
    # 4 negatives, 3 1 rels, 2 2 rels, 1 1 rel (if possible)
    print("Sampling qrels...")
    sample_per_score = {score: 4 - score for score in scores}
    for q_id in queries.keys():
        for score, n_samples in sample_per_score.items():
            qrels[q_id][score] = random.sample(qrels[q_id][score], k=min(n_samples, len(qrels[q_id][score])))

    ## for each query, sample the top-3 bm25 documents
    stemmer = Stemmer.Stemmer('english')

    # load, tokenize, and index corpus
    index_path = os.path.join('indexes', 'msmarco')
    if os.path.exists(index_path):
        print("Index found.  Loading retriever...")
        retriever = bm25s.BM25.load(index_path)
    else:
        print("Creating new Index...")
        retriever = bm25s.BM25()
        print("Loading corpus...")
        corpus = [text for _, text in dataset.docs_iter()]

        print("Processing corpus...")
        corpus_tok = bm25s.tokenize(corpus, stopwords='en', stemmer=stemmer)

        print("Indexing corpus...")
        retriever.index(corpus_tok)
        print("Saving index...")
        retriever.save(index_path)
        del corpus
        del corpus_tok

    print("Retrieving documents...")
    q_ids, q_texts = list(queries.keys()), list(queries.values())
    queries_tok = bm25s.tokenize(q_texts, stopwords='en', stemmer=stemmer)
    docs, scores = retriever.retrieve(queries_tok, k=20)
    for i, q_id in enumerate(q_ids):
        top_d_ids = list(map(str, docs[i].tolist()))
        new_d_ids = [d_id for d_id in top_d_ids if d_id not in nonzero_qrels[q_id]][:3]
        qrels[q_id][0].extend(new_d_ids)

    # for each query, sample 10 more random negatives
    for q_id in q_ids:
        rand_docids = np.random.choice(dataset.docs_count(), size=10+len(nonzero_qrels[q_id]), replace=False).tolist()
        rand_docids = list(map(str, rand_docids))
        rand_negs = [d_id for d_id in rand_docids if d_id not in nonzero_qrels[q_id]][:10]
        qrels[q_id][0].extend(rand_negs)

    # collect and retrieve all doc ids, and flatten qrels
    flat_qrels = {q_id: {} for q_id in q_ids}
    all_doc_ids = set()
    for q_id, scores_dict in qrels.items():
        for score, d_ids in scores_dict.items():
            all_doc_ids.update(d_ids)
            for d_id in d_ids:
                flat_qrels[q_id][d_id] = score

    # retrieve documents
    documents = {}
    docstore = dataset.docs_store()
    for d_id, doc_tuple in docstore.get_many(all_doc_ids).items():
        _, text = doc_tuple
        documents[d_id] = text

    print("Stopping and stemming queries and documents..")

    # tokenize texts
    documents = {d_id: word_tokenize(text) for d_id, text in documents.items()}
    queries = {q_id: word_tokenize(text) for q_id, text in queries.items()}

    # stop texts
    stopper = Stopper()
    documents_stopped = {d_id: stopper.stop(text) for d_id, text in documents.items()}
    queries_stopped = {q_id: stopper.stop(text) for q_id, text in queries.items()}

    # stem texts
    documents_stemmed = {d_id: stemmer.stemWords(text) for d_id, text in documents.items()}
    queries_stemmed = {q_id: stemmer.stemWords(text) for q_id, text in queries.items()}
    documents_stopped_stemmed = {d_id: stemmer.stemWords(words) for d_id, words in documents_stopped.items()}
    queries_stopped_stemmed = {q_id: stemmer.stemWords(words) for q_id, words in queries_stopped.items()}

    # save everything
    data = {
        "queries": queries,
        "queries_stopped": queries_stopped,
        "queries_stemmed": queries_stemmed,
        "queries_stopped_stemmed": queries_stopped_stemmed,
        "documents": documents,
        "documents_stopped": documents_stopped,
        "documents_stemmed": documents_stemmed,
        "documents_stopped_stemmed": documents_stopped_stemmed,
        "qrels": flat_qrels
    }
    with open(os.path.join('data', 'data.json'), 'w') as f:
        f.write(json.dumps(data))

    print("Done!")


if __name__ == "__main__":
    main()
