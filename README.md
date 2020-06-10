# CCKS-2020

- [CCKS 2020：新冠知识图谱构建与问答评测（四）新冠知识图谱问答评测
](https://biendata.com/competition/ccks_2020_7_4/)

### 0. 代码结构

- 主要有三大模块：
    1. dataset模块：数据预处理等与数据相关的
        1. config.py 所有文件路径写这里，方便查找
    2. models模块：各种深度学习模型
    3. qa模块：将训练好的模型和策略组合应用，完成问答  

**代码运行必须从几个接口模块开始；小脚本测试，可模仿接口模块，新写一个接口模块，在其中导入函数运行**

├── dao //数据库接口，准备做辅助缓存  
├── dataset //数据相关  
│ ├── data_helper.py //train数据准备，feed给模型    
│ ├── data_prepare.py //train数据预处理，将给定训练数据做各种处理，构造字典等  
│ └── kb_data_prepare.py //图谱数据预处理，将给定图谱做各种转换，构造字典等  
├── layers //损失函数等    
├── models //模型  
│ ├── entity_score.py // 对识别出的主实体打分模型    
│ ├── evaluate.py // 评价测试模块，载入训练好的模型，对输入做预测；提供一份模型载入的类封装    
│ ├── ner.py //主实体识别模型  
│ ├── relation_score.py //对实体关联的关系打分的模型    
│ └── trainer.py //训练模块；模型初始化到训练    
├── qa //问答模块   
│ ├── algorithms.py // 后处理算法     
│ ├── cache.py // 大文件，在单例模式缓存；避免多次载入内存；ent2id等放在这里，提供给其他模块公共使用    
│ ├── entity_score.py // 对识别出的主实体打分模型    
│ ├── evaluation_matrics.py // 指标计算      
│ ├── lac_tools.py // 分词模块自定义优化等     
│ ├── neo4j_graph.py //图数据库查询缓存等   
│ ├── qa.py //问答接口，将其他模块组装到这里完成问答     
│ ├── recognizer.py //主实体识别模块     
│ └── relation_extractor.py //实体关联关系识别     
├── utils //通用工具   
├── docs //文档   
├── examples //临时任务，模块试验等，单个脚本    
├── tests //测试  
├── config.py //所有数据路径和少量全局配置    
├── data.py //所有数据处理的入口文件    
├── manage.py //所有模型训练的入口文件    
├── qa.py //问答入口文件   
├── README.md //说明文档   
└── requirements.txt //依赖包  


### 1. 图数据库  
- 知识库管理系统
    - [gStore](http://gstore-pku.com/pcsite/index.html)
    - [gStore - github](https://github.com/pkumod/gStore/blob/master/docs/DOCKER_DEPLOY_CN.md)
- SPARQL  
    - 国际化资源标识符（Internationalized Resource Identifiers，简称IRI），与其相提并论的是URI（Uniform Resource Identifier，统一资源标志符）。  
        使用<uri>来表示一个IRI
    - Literal用于表示三元组中客体(Object)，表示非IRI的数据，例如字符串(String)，数字(xsd:integer)，日期(xsd:date)等。
        普通字符串等 "chat"
    - [RDF查询语言SPARQL - SimmerChan的文章 - 知乎
    ](https://zhuanlan.zhihu.com/p/32703794) 

neo4j数据库
- 安装  
    https://segmentfault.com/a/1190000015389941

- 运行 
    cd /home/wangshengguang/neo4j-community-3.4.5/bin
    ./neo4j start
    ./neo4j stop
    
- 数据导入   
    cd /home/wangshengguang/neo4j-community-3.4.5/bin  
    ./neo4j-admin import --database=graph.db --nodes /home/wangshengguang/ccks-2020/data/graph_entity.csv  --relationships /home/wangshengguang/ccks-2020/data/graph_relation.csv --ignore-duplicate-nodes=true --id-type INTEGER --ignore-missing-nodes=true  
 
- 创建索引  
CREATE CONSTRAINT ON (ent:Entity) ASSERT ent.id IS UNIQUE;  
CREATE INDEX ON :Entity(name)  
CREATE INDEX ON :Relation(name)  

 
- [neo4j学习笔记（三）——python接口-创建删除结点和关系](https://blog.csdn.net/qq_36591505/article/details/100987105)
- [neo4j︱与python结合的py2neo使用教程（四）](https://blog.csdn.net/sinat_26917383/article/details/79901207)
- [neo4j中文文档](http://neo4j.com.cn/public/docs/index.html)



## 2. 历年方案  
 
- [CCKS 2019 | 开放域中文KBQA系统 - 最AI的小PAI的文章 - 知乎
](https://zhuanlan.zhihu.com/p/92317079)
    1. 首先在句子中找到主题实体。在这里，我们使用了比赛组织者提供的Entity-mention文件和一些外部工具，例如paddle-paddle。
    2.  然后在关系识别模块中，通过提取知识图中的主题实体的子图来找到问题（也称为谓词）的关系。通过相似性评分模型获得所有关系的排名。
    3. 最后，在答案选择模块中，根据简单复杂问题分类器和一些规则，得出最终答案。  

- [百度智珠夺冠：在知识图谱领域百度持续领先 ](https://www.sohu.com/a/339187520_630344)
    1. 实体链接组件把问题中提及的实体链接到了知识库，并识别问题的核心实体。为了提高链接的精度，链接组件综合考虑了实体的子图与问题的匹配度、实体的流行度、指称正确度等多种特征，最后利用 LambdaRank 算法对实体进行排序，得到得分最高的实体。
    2. 子图排序组件目标是从多种角度计算问题与各个子图的匹配度，最后综合多个匹配度的得分，得到出得分最高的答案子图。
    3. 针对千万级的图谱，百度智珠团队采用了自主研发的策略来进行子图生成时的剪枝，综合考虑了召回率、精确率和时间代价等因素，从而提高子图排序的效率和效果。
        针对开放领域的子图匹配，采用字面匹配函数计算符号化的语义相似，应用 word2vec 框架计算浅层的语义匹配，最后应用 BERT 算法做深度语义对齐。
        除此之外，方案还针对具体的特征类型的问题进行一系列的意图判断，进一步提升模型在真实的问答场景中的效果和精度，更好地控制返回的答案类型，更符合真实的问答产品的需要。

- http://nlpprogress.com/

## 3. 数据分析及预处理  

### 3.1 原始数据 
原始数据存放在data/roaw_data 目录下
1. 问答数据 data\raw_data\ccks_2020_7_4_Data下
    1. 有标注训练集15999：data\raw_data\ccks_2020_7_4_Data\task1-4_train_2020.txt
    2. 无标注验证集（做提交）1529：data\raw_data\ccks_2020_7_4_Data\task1-4_valid_2020.questions
2. 图谱数据 data\raw_data\PKUBASE下
    1. 所有三元组66499745:data\raw_data\PKUBASE\pkubase-complete.txt

### 3.2 数据分析 
     三元组中实体分两类：<实体>和"属性"; 
     

### 3.3 预处理tips:   
    三元组中数据分两类，<实体>和"属性"; 
    预处理时将属性的双引号去掉(包括构建的字典和导入neo4j的数据全部双引号都被去掉了，主要考虑双引号作为json的key不方便保存)，方便使用；
    kb_data_prepare.py->iter_triples
    在最后提交时需要恢复 
    

## 4. 目前方案
    1. 先做NER识别 主实体
    2. 查找实体的关系，做分类，挑选出top 路径
    3. 生成sparql查询结果


### 4.1  主实体识别模块
    Recognizer


### 4.2 关系打分模块  
    RelationExtractor


### 1.3 后处理及查询模块
    Algorithms



## Others

- 本地端口映射后在本机访问远程图数据库  
    ssh -f wangshengguang@119.3.178.138 -N -L 7474:localhost:7474
    ssh -f wangshengguang@119.3.178.138 -N -L 7687:localhost:7687

    访问：http://localhost:7474/browser/



- [Jupyter Notebook 有哪些奇技淫巧？ - z.defying的回答 - 知乎
](https://www.zhihu.com/question/266988943/answer/1154607853)
- [linux下查看文件编码及修改编码](https://blog.csdn.net/jnbbwyth/article/details/6991425)

