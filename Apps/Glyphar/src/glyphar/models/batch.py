"""
Batch processing schemas for asynchronous OCR workloads.

Enables:
    - Queue-based processing (task status tracking)
    - Priority scheduling (priority field)
    - Failure isolation (per-task errors don't block batch)
    - Result aggregation (BatchResult)
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from .config import OCRConfig
from .output import OCROutput


class BatchStatus(str):
    """Valid states for batch task lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchTask(BaseModel):
    """
    Individual OCR task within a batch processing job.

    Represents unit of work — no execution logic. Status transitions:
        PENDING → RUNNING → COMPLETED/FAILED

    Design constraints:
        - Immutable after creation (task definition doesn't change)
        - Priority 0-10 (10 = highest urgency)
        - Error field populated only on FAILED status

    Example:
        >>> task = BatchTask(
        ...     task_id="task_001",
        ...     file_path="/docs/book.pdf",
        ...     config=OCRConfig(dpi=300),
        ...     priority=5
        ... )
    """

    task_id: str = Field(..., description="Unique task identifier")
    file_path: str = Field(..., description="Absolute path to input file")
    config: OCRConfig = Field(..., description="OCR configuration for this task")

    priority: int = Field(
        default=0, ge=0, le=10, description="Processing priority (0-10)"
    )
    status: str = Field(default=BatchStatus.PENDING, description="Current task state")
    error: Optional[str] = Field(None, description="Error message on failure")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = ConfigDict(extra="ignore", frozen=True)


class BatchResult(BaseModel):
    """
    Aggregated result of batch OCR processing job.

    Combines:
        - Task metadata (statuses, timing)
        - Successful OCR outputs (OCROutput list)
        - Derived metrics (success rate, duration)

    Usage pattern:
        1. Submit batch → receive BatchResult with PENDING tasks
        2. Poll status → tasks transition to RUNNING/COMPLETED/FAILED
        3. On completion → access results via .results property

    Example:
        >>> batch = BatchResult(
        ...     batch_id="batch_001",
        ...     tasks=[task1, task2, task3],
        ...     results=[output1, output2],  # task3 failed
        ...     started_at=datetime.now(),
        ...     finished_at=datetime.now()
        ... )
        >>> print(f"Success rate: {batch.success_rate:.1f}%")
    """

    batch_id: str = Field(..., description="Unique batch identifier")
    tasks: List[BatchTask] = Field(..., description="Individual task definitions")
    results: List[OCROutput] = Field(
        default_factory=list, description="Successful OCR outputs"
    )

    started_at: datetime = Field(..., description="Batch processing start timestamp")
    finished_at: Optional[datetime] = Field(
        None, description="Batch completion timestamp"
    )

    model_config = ConfigDict(extra="ignore", frozen=True)

    @property
    def total_tasks(self) -> int:
        """Total tasks submitted in batch."""
        return len(self.tasks)

    @property
    def completed_tasks(self) -> List[BatchTask]:
        """Tasks with COMPLETED status."""
        return [t for t in self.tasks if t.status == BatchStatus.COMPLETED]

    @property
    def failed_tasks(self) -> List[BatchTask]:
        """Tasks with FAILED status."""
        return [t for t in self.tasks if t.status == BatchStatus.FAILED]

    @property
    def success_rate(self) -> float:
        """Percentage of successfully completed tasks."""
        return len(self.completed_tasks) / self.total_tasks * 100 if self.tasks else 0.0

    @property
    def total_duration_s(self) -> Optional[float]:
        """Total batch processing duration in seconds."""
        return (
            (self.finished_at - self.started_at).total_seconds()
            if self.finished_at
            else None
        )
