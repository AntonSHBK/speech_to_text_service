from celery import Celery

from app.settings import settings


celery_app = Celery(
    "speech_to_text_service",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Помечает задачу как STARTED в хранилище результатов перед началом выполнения.
    task_track_started=True,
    # Публикует события жизненного цикла задач, чтобы Flower отображал активные,
    # зарезервированные и запланированные задачи.
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Явно включает повторные попытки подключения к брокеру при запуске
    # для совместимости с Celery 6.
    broker_connection_retry_on_startup=True,
    # Использует JSON для сериализации данных задач.
    task_serializer="json",
    # Использует JSON для сериализации результатов в хранилище.
    result_serializer="json",
    # Принимает от брокера только сообщения в формате JSON.
    accept_content=["json"],
    # Подтверждает выполнение задачи только после её завершения,
    # чтобы снизить вероятность потери задачи при сбое воркера.
    task_acks_late=True,
    # Возвращает задачу в очередь, если процесс воркера завершился
    # во время её обработки.
    task_reject_on_worker_lost=True,
    # Явно подтверждает завершившиеся с ошибкой или по таймауту задачи,
    # чтобы состояние брокера оставалось согласованным.
    task_acks_on_failure_or_timeout=True,
    # Резервирует по одной задаче на процесс воркера,
    # чтобы избежать голодания длительных задач.
    worker_prefetch_multiplier=1,
    # Сохраняет корневые обработчики логирования, настроенные в app.utils.logging,
    # чтобы трассировки Celery записывались в логи воркера.
    worker_hijack_root_logger=False,
    # Хранит результаты задач в хранилище в течение 24 часов.
    result_expires=86400,
    # Время (в секундах), после которого неподтверждённая задача
    # снова становится доступной в брокере.
    broker_transport_options={"visibility_timeout": 3600},
)

celery_app.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "cleanup.old_files",
        "schedule": max(settings.CLEANUP_INTERVAL_MINUTES, 1) * 60,
    },
}