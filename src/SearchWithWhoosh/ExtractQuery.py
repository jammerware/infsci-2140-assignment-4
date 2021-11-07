import Classes.Query as Query

class ExtractQuery:

    def __init__(self):
        # 1. you should extract the 4 queries from the Path.TopicDir
        # 2. the query content of each topic should be 1) tokenized, 2) to lowercase, 3) remove stop words, 4) stemming
        # 3. you can simply pick up title only for query.
        return

    # Return extracted queries with class Query in a list.
    def getQuries(self):
        queries=[]
        aQuery=Query.Query()
        aQuery.setTopicId("901")
        aQuery.setQueryContent("hong OR kong OR econom OR singapor")
        queries.append(aQuery)
        aQuery=Query.Query()
        aQuery.setTopicId("902")
        aQuery.setQueryContent("homosexu OR accept OR europ")
        queries.append(aQuery)
        aQuery=Query.Query()
        aQuery.setTopicId("903")
        aQuery.setQueryContent("star OR trek OR gener")
        queries.append(aQuery)
        aQuery=Query.Query()
        aQuery.setTopicId("904")
        aQuery.setQueryContent("progress OR dysphagia")
        queries.append(aQuery)

        return queries

