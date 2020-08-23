from mongoengine import Document, StringField, IntField, ListField


class Entity2id(Document):
    meta = {
        'db_alias': 'entity2id',
        'collection': 'graph',
        'strict': False,
        'indexes': [
            'name',
            'id'
        ]
    }

    name = StringField(required=True)
    id = IntField(required=True)


class Relation2id(Document):
    meta = {
        'db_alias': 'relation2id',
        'collection': 'graph',
        'strict': False,
        'indexes': [
            'name',
            'id'
        ]
    }

    name = StringField(required=True)
    id = IntField(required=True)


class Graph(Document):
    meta = {
        'db_alias': 'sub_graph',
        'collection': 'graph',
        'strict': False,
        'indexes': [
            'entity_name',
            'entity_id'
        ]
    }

    entity_name = StringField(required=True)
    entity_id = IntField(required=True)
    ins = ListField(required=True)
    outs = ListField(required=True)


class SubGraph(Document):
    meta = {
        'db_alias': 'sub_graph',
        'collection': 'graph',
        'strict': False,
        'indexes': [
            'entity_name',
            'entity_id'
        ]
    }

    entity_name = StringField(required=True)
    entity_id = IntField(required=True)
    sub_graph = ListField(required=True)
    hop = IntField(required=True)  # 几跳
