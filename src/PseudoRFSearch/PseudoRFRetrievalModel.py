from collections import Counter, namedtuple
from Classes.Document import Document
from Classes.Query import Query
from SearchWithWhoosh.QueryRetreivalModel import QueryRetrievalModel
from IndexingWithWhoosh.MyIndexReader import MyIndexReader

class PseudoRFRetreivalModel:
    def __init__(self, ixReader):
        self.indexReader: MyIndexReader = ixReader
        self.orig_model = QueryRetrievalModel(self.indexReader)
        self.collection_length = self.indexReader.get_collection_length()
        self.mu = 530

    # Search for the topic with pseudo relevance feedback.
    # The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
    # query: The query to be searched for.
    # TopN: The maximum number of returned document
    # TopK: The count of feedback documents
    # alpha: parameter of relevance feedback model
    # return TopN most relevant document, in List structure

    def retrieveQuery(self, query: Query, topN, topK, alpha):
        query_tokens = []
        for token in query.getQueryContent().split():
            # shortcutting stop words lol
            if token != 'OR' and self.indexReader.contains_token(token):
                query_tokens.append(token)

        # (1) you should first use the original retrieval model to get TopK documents, which will be regarded as feedback documents
        feedback_results = self.orig_model.retrieveQuery(query, topK)

        # get P(token|feedback documents)
        rf_scores = self.get_rf_scores(query_tokens, feedback_results)
            
        # cache the postings for quick lookup
        postings = { token: self.indexReader.getPostingList(token) for token in query_tokens}

        # key is the doc no, val is the final score
        # (we don't have to care which term generated which score)
        final_scores: dict[str, int] = {}

        # loop through each combination of document and query token, calculating its initial score
        # as a simple query likelihood probability and its final as a linear combination of the original
        # score and the feedback relevance value for the token
        for doc in feedback_results:
            doc_length = self.indexReader.getDocLength(doc.getDocId())
            final_scores[doc.docno] = 1

            for token in query_tokens:
                # original score is simple posterior probability
                token_count_in_doc = postings[token].get(doc.docno, 0)
                original_score = self.get_dirichlet_prior(token, token_count_in_doc, doc_length)
                if original_score == 0: print('orig 0', token)
                # final score given by a linear combination: alpha * Q|D + (1-alpha) Q|Feedbackdocs
                final_score = alpha * original_score + (1-alpha) * rf_scores[token]

                # (3) implement the relevance feedback model for each token: combine the each query token's original retrieval score P(token|document) with its score in feedback documents P(token|feedback model)
                # now that we know the final score of this token/doc combination, we can actually just multiply the current score 
                # # for this doc consecutively obtain the document's overall score across all query terms
                final_scores[doc.docno] *= final_score

        # update the feedback documents with their adjusted score (so that when the assignment prints,
        # it prints the post-RF score)
        for doc in feedback_results:
            doc.setScore(final_scores[doc.docno])

        # sort results by final score and retrieve top N
        sorted_results = sorted(feedback_results, key = lambda x: x.getScore(), reverse=True)
        return sorted_results[:topN]

    def get_dirichlet_prior(self, token: str, occurrences_in_document: int, doc_length: int):
        collection_probability = self.indexReader.CollectionFreq(token) / self.collection_length

        numerator = occurrences_in_document + (self.mu * collection_probability)
        denominator = doc_length + self.mu

        return numerator / denominator

    def get_rf_scores(self, query_tokens: list[str], feedback_results: list[Document]):
        # compound feedback docs into a single pseudo doc
        pseudo_doc = []
        for result in feedback_results:
            doc_tokens = self.indexReader.get_doc_content(result.docno).split()
            pseudo_doc += doc_tokens

        # get inputs needed for dirichlet prior
        doc_length = len(pseudo_doc)
        term_counts = Counter(pseudo_doc)

        # get smoothed P(query | pseudo-doc)
        rf_scores = {}
        for token in query_tokens:
            rf_scores[token] = self.get_dirichlet_prior(token=token, occurrences_in_document=term_counts[token], doc_length=doc_length)
                
        return rf_scores
