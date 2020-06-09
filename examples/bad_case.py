
'''

2020-06-09 00:32:53,712 - INFO qa.py run 59 - * get_most_overlap_path: ['<吉林大学>', '<校歌>'] ...
2020-06-09 00:32:53,712 - INFO neo4j_graph.py search_by_2path 83 - match (ent:Entity)-[r1:Relation]-(target) where ent.id=5199304  and r1.name='<校歌>' return DISTINCT target.name
2020-06-09 00:32:53,892 - INFO qa.py run 72 - * cypher answer: ['<吉林大学校歌>']
'''
