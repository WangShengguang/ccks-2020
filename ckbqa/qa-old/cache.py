from ckbqa.utils.decorators import singleton
from ckbqa.utils.tools import json_load
from config import Config


@singleton
class Memory(object):
    def __init__(self):
        self.entity2id = json_load(Config.entity2id)
        self.mention2entity = json_load(Config.mention2ent_json)
        self.all_attrs = set(json_load(Config.all_attrs_json))

    def get_entity_id(self, ent_name):
        if ent_name in self.entity2id:
            ent_id = self.entity2id[ent_name]
        else:
            # ent_name = ''.join(ent_patten.findall(ent_name))
            ent_id = self.entity2id.get(ent_name[1:-1], 0)
        return ent_id
