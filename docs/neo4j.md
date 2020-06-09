
## 安装启动
    D:\Program Files\neo4j-community-3.5.8\bin
    添加到path
    neo4j console

- [neo4j学习笔记（三）——python接口-创建删除结点和关系](https://blog.csdn.net/qq_36591505/article/details/100987105)
- [neo4j︱与python结合的py2neo使用教程（四）](https://blog.csdn.net/sinat_26917383/article/details/79901207)
- [neo4j中文文档](http://neo4j.com.cn/public/docs/index.html)

http://weikeqin.com/2017/04/11/neo4j-import/
https://blog.csdn.net/u013946356/article/details/82629014


cd neo4j-community-3.4.5/bin/  
./cypher-shell
neo4j start

ssh -f wangshengguang@119.3.178.138 -N -L 7474:localhost:7474
ssh -f wangshengguang@119.3.178.138 -N -L 7687:localhost:7687
http://localhost:7474/browser/


MATCH (n) DETACH DELETE n;




"""
CREATE CONSTRAINT ON (ent:Entity) ASSERT ent.id IS UNIQUE;
CREATE INDEX ON :Entity(name)
CREATE INDEX ON :Relation(name)

./neo4j-import --into /home/wangshengguang/neo4j-community-3.4.5/data/databases/graph.db --nodes /home/wangshengguang/ccks-2020/data/graph_entity.csv  --relationships /home/wangshengguang/ccks-2020/data/graph_relation.csv --trim-strings true --input-encoding UTF-8 --id-type INTEGER --stacktrace true --bad-tolerance 0 --skip-bad-relationships true --skip-duplicate-nodes false


./neo4j-admin import --database=graph.db --nodes /home/wangshengguang/ccks-2020/data/graph_entity.csv  --relationships /home/wangshengguang/ccks-2020/data/graph_relation.csv --ignore-duplicate-nodes=true --id-type INTEGER --ignore-missing-nodes=true  


"""