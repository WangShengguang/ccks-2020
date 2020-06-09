# CCKS-2020

- [CCKS 2020：新冠知识图谱构建与问答评测（四）新冠知识图谱问答评测
](https://biendata.com/competition/ccks_2020_7_4/)

- [RDF查询语言SPARQL - SimmerChan的文章 - 知乎
](https://zhuanlan.zhihu.com/p/32703794)  


## 知识库管理系统
- [gStore](http://gstore-pku.com/pcsite/index.html)
- [gStore - github](https://github.com/pkumod/gStore/blob/master/docs/DOCKER_DEPLOY_CN.md)


http://nlpprogress.com/

## SPARQL  
    - 国际化资源标识符（Internationalized Resource Identifiers，简称IRI），与其相提并论的是URI（Uniform Resource Identifier，统一资源标志符）。  
        使用<uri>来表示一个IRI
    - Literal用于表示三元组中客体(Object)，表示非IRI的数据，例如字符串(String)，数字(xsd:integer)，日期(xsd:date)等。
        普通字符串等 "chat"

## KBQA相关论文
 
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

## 涉及技术

- 实体识别
    分词工具对比，百度lac最好。  
    https://github.com/hankcs/HanLP  
    https://github.com/baidu/lac  
    
- 实体链接 
    - [知识图谱 | 实体链接 - 安小飞的文章 - 知乎](https://zhuanlan.zhihu.com/p/81073607)
    - [【知识图谱】实体链接：一份“由浅入深”的综述 - Nicolas的文章 - 知乎](https://zhuanlan.zhihu.com/p/100248426)



### 方案一 
    1. 先做NER识别 主实体
    2. 


centos javac 
https://segmentfault.com/a/1190000015389941


## 临时环境变量设置
export PATH=$PATH:.

本地端口映射 
ssh -f wangshengguang@119.3.178.138  -N -L 9911:localhost:9911
[Jupyter Notebook 有哪些奇技淫巧？ - z.defying的回答 - 知乎
](https://www.zhihu.com/question/266988943/answer/1154607853)



[linux下查看文件编码及修改编码](https://blog.csdn.net/jnbbwyth/article/details/6991425)

