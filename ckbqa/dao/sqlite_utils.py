from .db import DB
from .db_tools import try_commit_rollback, try_commit_rollback_expunge
from .sqlite_models import Entity2id, SubGraph
from ..utils.tools import singleton


@singleton
class SqliteDB(DB):

    @try_commit_rollback
    def get_id_by_entity_name(self, *, entity_name, pure_name):
        sql = 'select entity_name from '
        ws_video = self.select(sql)
        return ws_video

    @try_commit_rollback_expunge
    def get_subGraph_by_entity_ids(self, entity_ids):
        # sub_graph = self.session.query(SubGraph).filter_by(id=entity_id).first()
        sub_graph = self.session.query(SubGraph).filter(SubGraph.id.in_(entity_ids))
        return sub_graph
