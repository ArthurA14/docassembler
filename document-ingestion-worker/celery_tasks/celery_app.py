# celery_tasks/celery_app.py
import os

from celery import Celery

celery_app = Celery(
    "doc_assemblererator",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
    include=["celery_tasks.importation"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="celery",
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "432000")),
    task_routes={
        "celery_tasks.importation.import_documents_task": {"queue": "importation"},
    },
)

# celery_app.autodiscover_tasks(["celery_tasks"])
# import celery_tasks.importation
