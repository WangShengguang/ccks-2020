from .db import Mongo
from .mongo_models import Graph, Entity2id


class MongoDB(Mongo):
    def entity2id(self, entity_name):
        return Entity2id.objects(name_eq=entity_name).all()

    def save_graph(self, graph_node: Graph):
        return Graph.objects(graph_node).all()
