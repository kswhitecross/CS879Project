import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from typing import Iterable
import pytrec_eval


class BagOfWordsEmbedding:
    """
    This class maps lists of tokens into bag-of-words vectors (with get_embeddings), and then converts the resultant
     per-word-scores into dictionaries mapping the words themselves into their contribution to the score.
    """
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def get_embeddings(self, query_texts: list[list[str]], document_texts: list[list[str]],
                       ) -> tuple[np.ndarray, np.ndarray]:
        query_texts = [" ".join(toks) for toks in query_texts]
        docuemnt_texts = [" ".join(toks) for toks in document_texts]
        all_texts = query_texts + docuemnt_texts
        x = self.vectorizer.fit_transform(all_texts).toarray()
        return x[:len(query_texts), :], x[len(query_texts):, :]

    def vectors_to_word_scores(self, bow_scores: Iterable[np.ndarray], include_words: list[list[str]]=None) -> list[dict[str, float]]:
        """
        Maps score vectors to dictionaries mapping the words with nonzero scores to scores.  If include_words is not
         None, then each word in include_words[i] will also be included in output[i] with a score of 0 (unless it has a
         non-zero score in bow_scores)
        """
        words_map = self.vectorizer.get_feature_names_out()
        maps = []
        for i,bow_score in enumerate(bow_scores):
            words = words_map[bow_score.nonzero()].tolist()
            scores = bow_score[bow_score.nonzero()].tolist()
            if include_words is not None:
                word_map = {word: 0 for word in include_words[i]}
            else:
                word_map = {}
            word_map.update({word: score for word, score in zip(words, scores)})
            maps.append(word_map)
        return maps


class ScoredDocument:
    def __init__(self, d_id: str, score: float, word_scores: dict[str, float],
                 missing_word_scores: dict[str, float]=None):
        """
        Stores a single scored document with id `d_id`, score `score` and `word_scores`, a dictionary mapping unique
         word tokens to their contributions to the score.  Optionally, `missing_word_scores` is a dict which maps words
         not present in the document to their contribution to the score.
        """
        self.d_id = d_id
        self.score = score
        self.word_scores = word_scores
        self.missing_word_scores = missing_word_scores


class RetrievalModelScores:
    def __init__(self,
                 doc_scores: dict[str, dict[str, ScoredDocument]]):
        """
        Stores the detailed output of a retrieval model. `doc_scores` is a dictionary mapping query_ids to dictionaries
         mapping document_ids to Scored documents
        """
        self.doc_scores = doc_scores

        # compute document ranks
        self.doc_ranks = {}
        for q_id, scoredocs in self.doc_scores.items():
            pairs = [(d_id, doc.score) for d_id, doc in scoredocs.items()]
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.doc_ranks[q_id] = {d_id: rank for (d_id, _), rank in zip(pairs, range(1, len(pairs) + 1))}

    def compute_metrics(self, qrels: dict[str, dict[str, int]], metrics: dict=None) -> dict[str, dict[str, float]]:
        """
        Uses pytrec_eval to compute the requested metrics.  If `metrics` is None then default metrics will be computed
         instead.
        """
        # default metrics
        if metrics is None:
            metrics = {
                'P.10',
                'recall.10',
                'map',
                'ndcg_cut.10'
            }

        # build doc scores dict for pytrec eval
        scores = {q_id: {d_id: doc.score for d_id, doc in q_doc_scores.items()} for q_id, q_doc_scores in self.doc_scores.items()}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
        return evaluator.evaluate(scores)


def compute_tf(queries: dict[str, list[str]], documents: dict[str, list[str]]) -> RetrievalModelScores:
    """
    Computes term frequency retrieval model scores for each document for each query.
    """
    # get texts
    query_texts = [text for text in queries.values()]
    doc_texts = [text for text in documents.values()]

    # embed each query and document
    embedding = BagOfWordsEmbedding()
    query_embs, doc_embs = embedding.get_embeddings(query_texts, doc_texts)

    # shape (Q, D, vocab)
    per_word_scores = query_embs[:, np.newaxis, :] * doc_embs[np.newaxis, :, :]

    # shape (Q, D)
    scores = per_word_scores.sum(axis=-1).tolist()

    # build retrieval model scores
    doc_scores = {}
    for i, query_id in enumerate(queries.keys()):
        doc_scores[query_id] = {}
        # get word score maps
        # include all terms in doc texts
        word_score_maps = embedding.vectors_to_word_scores(per_word_scores[i], doc_texts)
        for j, doc_id in enumerate(documents.keys()):
            doc_scores[query_id][doc_id] = ScoredDocument(doc_id, scores[i][j], word_score_maps[j], {})

    return RetrievalModelScores(doc_scores)


def compute_bm25(queries: dict[str, list[str]], documents: dict[str, list[str]], k_1: float = 1.2, b: float = 0.75) -> RetrievalModelScores:
    """
    Computes BM25 retrieval model scores for each document for each query.
    """
    # get texts
    query_texts = [text for text in queries.values()]
    doc_texts = [text for text in documents.values()]

    # embed each query and document
    embedding = BagOfWordsEmbedding()
    query_embs, doc_embs = embedding.get_embeddings(query_texts, doc_texts)
    N = len(doc_texts)

    # compute collection frequencies (# of docs a term appears in) and IDF
    C = doc_embs.astype(bool).sum(axis=0)
    IDF = np.log((N - C + 0.5) / (C + 0.5))  # (n_vocab, )

    # compute document lengths
    dl = doc_embs.sum(axis=1)  # (n_d, )
    avg_dl = dl.mean()
    dl_norm = k_1 * (1 - b + b * (dl / avg_dl))  # (n_d, )
    eps = 1e-6
    denominator = dl_norm[:, np.newaxis] + doc_embs + eps  # (n_d, n_vocab)

    # compute numerator
    numerator = (k_1 + 1) * query_embs[:, np.newaxis, :] * doc_embs[np.newaxis, :, :]  # (n_q, n_d, n_vocab)

    # compute bm25
    per_word_bm25 = numerator / denominator[np.newaxis, :, :] * IDF[np.newaxis, np.newaxis, :]  # (n_q, n_d, n_vocab)
    per_doc_scores = per_word_bm25.sum(axis=-1).tolist()  # (n_q, n_d)

    # build retrieval model scores
    doc_scores = {}
    for i, query_id in enumerate(queries.keys()):
        doc_scores[query_id] = {}
        # get word score maps
        # include all terms in doc texts
        word_score_maps = embedding.vectors_to_word_scores(per_word_bm25[i], doc_texts)
        for j, doc_id in enumerate(documents.keys()):
            doc_scores[query_id][doc_id] = ScoredDocument(doc_id, per_doc_scores[i][j], word_score_maps[j], {})

    return RetrievalModelScores(doc_scores)


def compute_bm25_range(queries: dict[str, list[str]], documents: dict[str, list[str]], k_1_range: Iterable[float],
                       b_range: Iterable[float]) -> dict[float, dict[float, RetrievalModelScores]]:
    """
    Computes the per-word bm25 scores for a range of k_1, and b values, specified by b_range and k_1_range.  Returns a
     dictionary mapping k_1 values to dictionaries mapping b values to RetrievalModelScores objects that result from
     those parameter choices
    """
    ret_dict = {}
    for k_1_val in k_1_range:
        ret_dict[k_1_val] = {}
        for b_val in b_range:
            ret_dict[k_1_val][b_val] = compute_bm25(queries, documents, k_1_val, b_val)
    return ret_dict


def compute_ql(queries: dict[str, list[str]], documents: dict[str, list[str]], lambda_p: float = 0.2):
    """
    Computes per-word QL scores for each query for each word in the document.  Here, the words that are present in the
     query but not present in the document play a role in the score as well, and will be set in each ScoredDocument's
     missing_word_scores
    """
    # get texts
    query_texts = [text for text in queries.values()]  # (n_q, n_vocab)
    doc_texts = [text for text in documents.values()]  # (n_d, n_vocab)

    # embed each query and document
    embedding = BagOfWordsEmbedding()
    query_embs, doc_embs = embedding.get_embeddings(query_texts, doc_texts)

    # get doc length, and collection length
    dl = doc_embs.sum(axis=-1)  # (n_d)
    C = dl.sum()

    # get doc_mle and collec_mle parameters
    doc_mle = (1 - lambda_p) * doc_embs / dl[:, np.newaxis]  # (n_d, n_vocab)
    collec_mle = lambda_p * doc_embs.sum(axis=0) / C  # (n_vocab)

    # indicator function for terms not in the document
    doc_present_words = doc_embs.astype(bool)
    doc_missing_words = ~doc_present_words

    eps = 1e-6

    # per_word_score for words in both document and query
    # (n_q, n_d, n_vocab)
    present_word_scores = doc_present_words[np.newaxis, :, :] * (query_embs[:, np.newaxis, :] *
                           np.log(doc_mle[np.newaxis, :, :] + collec_mle[np.newaxis, np.newaxis, :] + eps))

    # per_word_score for words in the query and not in the document
    # (n_q, n_d, n_vocab)
    missing_word_scores = (query_embs[:, np.newaxis, :] * doc_missing_words[np.newaxis, :, :] *
                           np.log(collec_mle[np.newaxis, np.newaxis, :] + eps))

    per_doc_scores = (present_word_scores + missing_word_scores).sum(axis=-1).tolist()  # (n_q, n_d)

    # build retrieval model scores
    doc_scores = {}
    for i, query_id in enumerate(queries.keys()):
        doc_scores[query_id] = {}
        # get word score maps
        # include all terms in doc texts
        present_word_maps = embedding.vectors_to_word_scores(present_word_scores[i], doc_texts)
        missing_word_maps = embedding.vectors_to_word_scores(missing_word_scores[i])
        for j, doc_id in enumerate(documents.keys()):
            doc_scores[query_id][doc_id] = ScoredDocument(doc_id, per_doc_scores[i][j], present_word_maps[j],
                                                          missing_word_maps[j])

    return RetrievalModelScores(doc_scores)


def compute_ql_range(queries: dict[str, list[str]], documents: dict[str, list[str]], lambda_range: Iterable[float]):
    """
    Returns a dict mapping each lambda value in lambda range to its resultant RetrievalModelScores computed from
     compute_QL with that lambda value.
    """
    ret_dict = {}
    for lambda_p in lambda_range:
        ret_dict[lambda_p] = compute_ql(queries, documents, lambda_p)
    return ret_dict


# testing and debugging
if __name__ == "__main__":
    import json
    import os
    with open('data/data.json') as f:
        data = json.loads(f.read())
    queries = data['queries_stopped_stemmed']
    documents = data['documents_stopped_stemmed']

    ql_results = compute_ql(queries, documents, 0.2)

    print('Done!')