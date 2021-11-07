from whoosh.matching.mcore import Matcher
import Classes.Path as Path
import whoosh.index as index
from whoosh.index import FileIndex
from whoosh.reading import IndexReader
from whoosh.searching import Searcher
from whoosh.query import *
from whoosh.qparser import QueryParser
from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED

# Efficiency and memory cost should be paid with extra attention.


class MyIndexReader:

    searcher = []

    def __init__(self, type):
        path_dir = Path.IndexWebDir
        if type == "trectext":
            path_dir = Path.IndexTextDir

        self.index: FileIndex = index.open_dir(path_dir)
        self.searcher: Searcher = self.index.searcher()
        self.reader: IndexReader = self.index.reader()
        self.all_terms = list(self.reader.field_terms('doc_content'))

    # Return the integer DocumentID of input string DocumentNo.
    def getDocId(self, docNo):
        return self.searcher.document_number(doc_no=docNo)

    # Return the string DocumentNo of the input integer DocumentID.
    def getDocNo(self, docId):
        return self.searcher.stored_fields(docId)["doc_no"]

    # Return DF.
    def DocFreq(self, token):
        return self.reader.doc_frequency('doc_content', token)
        
    # Return the frequency of the token in whole collection/corpus.
    def CollectionFreq(self, token):
        return self.reader.frequency('doc_content', token)

    # Return posting list in form of {documentID:frequency}.
    def getPostingList(self, token):
        results = self.searcher.search(Term("doc_content", token), limit=None)
        postList = {}
        for result in results:
            words = self.searcher.stored_fields(result.docnum)["doc_content"].split(" ")
            count = 0
            for word in words:
                if word == token:
                    count += 1
            postList[result.docnum] = count
        return postList

    # Return the length of the requested document.
    def getDocLength(self, docId):
        return self.reader.doc_field_length(docId, 'doc_content')

    def get_docs_by_tokens(self, tokens: list[str]):
        terms = [Term("doc_content", token) for token in tokens]
        results = self.searcher.search(Or(terms), limit=None)
        
        return results

    def get_doc_content(self, doc_no) -> str:
        return self.searcher.document(doc_no=doc_no)['doc_content']

    def get_token_probability(self, token):
        return self.CollectionFreq(token) / self.index.field_length('doc_content')

    def get_collection_length(self):
        return self.index.field_length('doc_content')

    def contains_token(self, token):
        return token in self.all_terms

    def total_doc_count(self):
        return len(list(self.reader.all_doc_ids()))