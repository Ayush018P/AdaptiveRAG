from app.core.config import settings
from app.models.database import SessionLocal
from app.models.configuration import Configuration
from app.models.workspace import Workspace
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    def _ensure_initial_config(self, workspace_id: str, db: SessionLocal):
        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not ws:
            ws = Workspace(id=workspace_id, name=f"Workspace {workspace_id}")
            db.add(ws)
            db.commit()

        active = db.query(Configuration).filter(
            Configuration.workspace_id == workspace_id,
            Configuration.status == "ACTIVE"
        ).first()
        
        if not active:
            last_conf = db.query(Configuration).filter(
                Configuration.workspace_id == workspace_id
            ).order_by(Configuration.version_id.desc()).first()
            v_id = 1 if not last_conf else last_conf.version_id + 1
            
            new_conf = Configuration(
                version_id=v_id,
                workspace_id=workspace_id,
                top_k=settings.top_k,
                dense_weight=settings.dense_weight,
                bm25_weight=settings.bm25_weight,
                reranking_enabled=settings.enable_reranker,
                query_rewrite_enabled=getattr(settings, 'query_rewrite_enabled', False),
                status="ACTIVE",
                source_experiment="System Initialization"
            )
            db.add(new_conf)
            db.commit()
            return new_conf
        return active
            
    def get_active_config(self, workspace_id: str, db=None):
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True
        try:
            active = self._ensure_initial_config(workspace_id, db)
            return {
                "top_k": active.top_k,
                "dense_weight": active.dense_weight,
                "bm25_weight": active.bm25_weight,
                "enable_reranker": active.reranking_enabled,
                "query_rewrite_enabled": active.query_rewrite_enabled
            }
        except Exception:
            if should_close:
                db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    def activate_config(self, workspace_id: str, new_config: dict, source_experiment: str = None, is_experimental: bool = False, db=None):
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True
        try:
            if not is_experimental:
                active = db.query(Configuration).filter(
                    Configuration.workspace_id == workspace_id,
                    Configuration.status == "ACTIVE"
                ).first()
                if active:
                    active.status = "SUPERSEDED"
            
            last_conf = db.query(Configuration).filter(
                Configuration.workspace_id == workspace_id
            ).order_by(Configuration.version_id.desc()).first()
            v_id = 1 if not last_conf else last_conf.version_id + 1
            
            # Fetch base defaults to fallback on
            base_config = self._ensure_initial_config(workspace_id, db)
            
            conf_model = Configuration(
                version_id=v_id,
                workspace_id=workspace_id,
                top_k=new_config.get("top_k", base_config.top_k),
                dense_weight=new_config.get("dense_weight", base_config.dense_weight),
                bm25_weight=new_config.get("bm25_weight", base_config.bm25_weight),
                reranking_enabled=new_config.get("enable_reranker", base_config.reranking_enabled),
                query_rewrite_enabled=new_config.get("query_rewrite_enabled", base_config.query_rewrite_enabled),
                status="EXPERIMENTAL" if is_experimental else "ACTIVE",
                source_experiment=source_experiment
            )
            db.add(conf_model)
            db.commit()
            return v_id
        except Exception:
            if should_close:
                db.rollback()
            raise
        finally:
            if should_close:
                db.close()

    def rollback_config(self, workspace_id: str, version_id: int = None, db=None):
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True
        try:
            if version_id is None:
                # Rollback from experimental to last ACTIVE (this is basically a no-op if we don't mutate global state)
                # Just keeping the method signature for compatibility. 
                return
                
            target = db.query(Configuration).filter(
                Configuration.workspace_id == workspace_id,
                Configuration.version_id == version_id
            ).first()
            if not target:
                raise ValueError(f"Configuration version {version_id} not found in workspace {workspace_id}.")
            
            if target.status == "ACTIVE":
                return # Already active
                
            current_active = db.query(Configuration).filter(
                Configuration.workspace_id == workspace_id,
                Configuration.status == "ACTIVE"
            ).first()
            if current_active:
                current_active.status = "SUPERSEDED"
                
            last_conf = db.query(Configuration).filter(
                Configuration.workspace_id == workspace_id
            ).order_by(Configuration.version_id.desc()).first()
            v_id = 1 if not last_conf else last_conf.version_id + 1
            
            new_conf = Configuration(
                version_id=v_id,
                workspace_id=workspace_id,
                top_k=target.top_k,
                dense_weight=target.dense_weight,
                bm25_weight=target.bm25_weight,
                reranking_enabled=target.reranking_enabled,
                query_rewrite_enabled=target.query_rewrite_enabled,
                status="ACTIVE",
                source_experiment=f"Rollback to v{version_id}"
            )
            db.add(new_conf)
            db.commit()
            return v_id
        except Exception:
            if should_close:
                db.rollback()
            raise
        finally:
            if should_close:
                db.close()

config_manager = ConfigurationManager()
