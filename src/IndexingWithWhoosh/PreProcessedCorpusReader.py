import Classes.Path as Path

class PreprocessedCorpusReader:

    corpus = 0

    def __init__(self, type):
        self.corpus = open(Path.ResultHM1 + type, "r", encoding="utf8")

    def nextDocument(self):
        docNo=self.corpus.readline().strip()
        if docNo=="":
            self.corpus.close()
            return
        content=self.corpus.readline().strip()
        return [docNo, content]