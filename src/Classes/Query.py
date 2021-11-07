class Query:

    def __init__(self):
        return

    queryContent = ""
    topicId = ""

    def getQueryContent(self):
        return self.queryContent

    def getTopicId(self):
        return self.topicId

    def setQueryContent(self, content):
        self.queryContent=content

    def setTopicId(self, id):
        self.topicId=id