# -*-coding:utf-8 -*-

from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import CHAR, INTEGER, TEXT, SMALLINT
from sqlalchemy.ext.declarative import declarative_base

BaseModel = declarative_base()  # 创建一个类，这个类的子类可以自动与一个表关联


class Graph(BaseModel):
    __tablename__ = 'graph'

    id = Column(INTEGER, primary_key=True, comment="entity id")
    name = Column(CHAR(256), default=None, index=True, unique=True, comment="entity name")
    pure_name = Column(CHAR(256), default=None, index=True, comment="entity name")
    in_ids = Column(TEXT, default=[], comment="入度")
    out_ids = Column(TEXT, default=[], comment="出度")
    type = Column(SMALLINT, default=0, comment="entity:1, relation:2")


class SubGraph(BaseModel):
    __tablename__ = 'sub_graph'

    id = Column(INTEGER, primary_key=True, comment="entity id")
    name = Column(CHAR(256), default=None, index=True, unique=True, comment="entity name")
    pure_name = Column(CHAR(256), default=None, index=True, comment="entity name")
    sub_graph_ids = Column(TEXT, default={}, comment="子图")
    sub_graph_names = Column(TEXT, default={}, comment="子图")
    type = Column(SMALLINT, index=True, nullable=False, comment="entity:1, relation:2")


class Entity2id(BaseModel):
    __tablename__ = 'entity2id'

    entity_id = Column(INTEGER, default=None, primary_key=True, comment="entity id")
    entity_name = Column(CHAR(256), nullable=False, index=True, unique=True, comment="entity name")
    pure_name = Column(CHAR(256), nullable=False, index=True, comment="entity name")
