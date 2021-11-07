import IndexingWithWhoosh.MyIndexReader as MyIndexReader
import SearchWithWhoosh.QueryRetreivalModel as QueryRetreivalModel
import SearchWithWhoosh.ExtractQuery as ExtractQuery
import datetime


startTime = datetime.datetime.now()
index = MyIndexReader.MyIndexReader("trectext")
search = QueryRetreivalModel.QueryRetrievalModel(index)
extractor = ExtractQuery.ExtractQuery()
queries= extractor.getQuries()
for query in queries:
    print(query.topicId,"\t",query.queryContent)
    results = search.retrieveQuery(query, 20)
    rank = 1
    for result in results:
        print(query.getTopicId()," Q0 ",result.getDocNo(),' ',rank," ",result.getScore()," MYRUN",)
        rank +=1

endTime = datetime.datetime.now()
print ("query search time: ", endTime - startTime)