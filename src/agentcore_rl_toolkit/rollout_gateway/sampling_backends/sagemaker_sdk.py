"""``SageMakerSdkBackend`` — in-process sampling via the SageMaker Training Sessions SDK.

Wraps a SageMaker ``SamplingClient``; rendering is done by the gateway (same model
as :class:`TinkerSdkBackend` — the SageMaker backend cannot render itself). This backend
only samples ``token_ids -> token_ids + logprobs``.

Unlike Tinker (which takes a fixed client), the SageMaker client needs to be rebound after
every weight update (``TrainingClient.save_weights_and_get_sampling_client()`` returns
a new client each time, and reusing the old one silently samples stale weights).
``set_sampling_client()`` handles this atomically while the gateway's background thread
may already be in a ``generate()`` call — in-flight requests finish against the old
weights, all subsequent calls use the new client.

Each call awaits ``APIFuture.result_async()``, which holds a worker thread only for
each status poll (one blocking ``Get*OperationResult`` per ``to_thread``), not for the
whole operation; blocking ``result()`` in ``to_thread`` held one of the stock
``min(32, cpu + 4)`` threads per in-flight sample, queueing samples behind each other at
hundreds of concurrent rollouts. The polls run on the event loop's default executor,
which the backend sizes to ``poll_threads`` the first time it samples on a loop.

The service rejects ``maxTokens`` below 512. ``min_new_tokens`` declares that floor so
the gateway adapter treats a context budget with less room than that as exhausted
(``context_length_exceeded``) instead of sending a request the service refuses.

``sagemaker`` is imported lazily so the core gateway stays lean. Install the SDK wheel
(``pip install sagemaker_train-*.whl``) before using this backend.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sagemaker.train.training_session import SamplingParams as SagemakerSamplingParams
from sagemaker.train.training_session import StopReason

from ..trajectory import TurnRecord

logger = logging.getLogger(__name__)


class SageMakerSdkBackend:
    """``SamplingBackend`` over a SageMaker Training Sessions ``SamplingClient``.

    Args:
        sampling_client: A SageMaker ``SamplingClient`` obtained from
            ``TrainingClient.create_sampling_client()`` or
            ``TrainingClient.save_weights_and_get_sampling_client()``.
            Weights are fixed at client creation — call ``set_sampling_client()``
            after every ``save_weights_and_get_sampling_client()`` to stay on-policy.
        poll_threads: Size of the default executor installed on the loop running
            ``generate()`` (it runs the SDK's status polls). ``None`` keeps the
            loop's executor as is.
    """

    # SamplingParams.maxTokens floor in the Training Sessions API (min=512 in the service model).
    min_new_tokens = 512

    def __init__(
        self,
        sampling_client: Any,
        poll_threads: int | None = 256,
    ) -> None:
        self._sampling_client = sampling_client
        self._lock = threading.Lock()
        self._poll_threads = poll_threads
        self._executor_loop: asyncio.AbstractEventLoop | None = None

    def set_sampling_client(self, sampling_client: Any) -> None:
        """Rebind to a new ``SamplingClient`` (e.g. after a weight update).

        Thread-safe: the training thread can call this while the gateway's background
        thread may be mid-generate. In-flight calls finish against the old client; all
        subsequent ``generate()`` calls acquire the new one.
        """
        with self._lock:
            self._sampling_client = sampling_client

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict,
        session_id: str | None = None,
        image_data: Any = None,
        video_data: Any = None,
    ) -> TurnRecord:
        with self._lock:
            sc = self._sampling_client

        sp = SagemakerSamplingParams(
            max_tokens=int(sampling_params.get("max_new_tokens", 4096)),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=sampling_params.get("top_k"),
            stop=sampling_params.get("stop") or [],
        )
        timeout = float(sampling_params.get("timeout", 900.0))

        loop = asyncio.get_running_loop()
        if self._poll_threads is not None and self._executor_loop is not loop:
            loop.set_default_executor(
                ThreadPoolExecutor(max_workers=self._poll_threads, thread_name_prefix="sagemaker-poll")
            )
            self._executor_loop = loop

        op = sc.sample(prompt=list(prompt_ids), num_samples=1, sampling_params=sp)
        result = await op.result_async(timeout=timeout)

        seq = result.sequences[0]
        output_ids = list(seq.tokens)
        output_log_probs = list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else []
        finish = "length" if seq.stop_reason == StopReason.LENGTH else "stop"

        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=output_log_probs,
        )


__all__ = ["SageMakerSdkBackend"]
