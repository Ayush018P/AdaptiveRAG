from sqlalchemy.orm import Session
from app.models.experiment import Experiment

class MemoryEngine:
    def find_previous_success(self, workspace_id: str, failure_type: str, db: Session):
        # Look for the best recent accepted experiment for this failure type
        experiment = db.query(Experiment)\
            .filter(Experiment.workspace_id == workspace_id)\
            .filter(Experiment.failure_type == failure_type)\
            .filter(Experiment.decision == "ACCEPT")\
            .order_by(Experiment.score_delta.desc())\
            .first()
            
        if experiment:
            import json
            return {
                "found": True,
                "experiment_id": experiment.id,
                "hypothesis": json.loads(experiment.hypothesis),
                "score_delta": experiment.score_delta
            }
        return {"found": False}

memory_engine = MemoryEngine()
