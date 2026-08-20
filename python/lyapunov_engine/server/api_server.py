import os
import time
import uuid
import asyncio
from typing import Optional, AsyncGenerator, List
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lyapunov_engine.engine.sampling import SamplingParams
from lyapunov_engine.engine.llm_engine import LLMEngine, EngineConfig
from lyapunov_engine.engine.entropy import compute_semantic_entropy
from lyapunov_engine.engine.lyapunov import compute_lyapunov_divergence
from lyapunov_engine.server.protocol import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ModelCard,
    ModelList,
    UsageInfo,
    StabilityDiagnostics,
)


def run_stability_pipeline(
    engine: LLMEngine,
    tokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 64
) -> StabilityDiagnostics:
    """Run dynamic safety-gated stability diagnostics (Lyapunov + Semantic Entropy)."""
    context_len = len(prompt_tokens)
    free_gb = 16.0
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(engine.device)
        free_gb = free_bytes / (1024 ** 3)

    # Tier 3: Exceeds safe context window or low memory headroom
    if context_len > 128000 or free_gb < 2.0:
        return StabilityDiagnostics(
            lyapunov_exponent=0.0,
            semantic_entropy=0.0,
            confidence_rating="high",
            diagnostics_status="disabled_context_limit",
            cluster_count=1
        )

    # Tier 2: 64k < context_len <= 128k (Lightweight mode: N=2 paths, skip Lyapunov)
    if context_len > 64000:
        candidates = []
        for _ in range(2):
            req_id = f"diag_{uuid.uuid4()}"
            sp = SamplingParams(temperature=0.7, max_tokens=min(max_tokens, 32))
            engine.add_request(req_id, prompt_tokens, sp)
            out_tokens = []
            while engine.has_unfinished_requests():
                step_outs = engine.step()
                for o in step_outs:
                    if o.request_id == req_id:
                        out_tokens = o.output_tokens
            candidates.append(tokenizer.decode(out_tokens, skip_special_tokens=True))

        entropy, conf_label, clusters = compute_semantic_entropy(candidates)
        return StabilityDiagnostics(
            lyapunov_exponent=0.0,
            semantic_entropy=entropy,
            confidence_rating=conf_label,
            diagnostics_status="lightweight",
            cluster_count=len(set(clusters)) if clusters else 1
        )

    # Tier 1: S <= 64k (Full mode: Lyapunov sensitivity + N=4 Semantic Entropy paths)
    lambda_exp, _ = compute_lyapunov_divergence(
        engine.model, prompt_tokens, num_steps=8, device=str(engine.device)
    )

    candidates = []
    for _ in range(4):
        req_id = f"diag_{uuid.uuid4()}"
        sp = SamplingParams(temperature=0.7, max_tokens=min(max_tokens, 32))
        engine.add_request(req_id, prompt_tokens, sp)
        out_tokens = []
        while engine.has_unfinished_requests():
            step_outs = engine.step()
            for o in step_outs:
                if o.request_id == req_id:
                    out_tokens = o.output_tokens
        candidates.append(tokenizer.decode(out_tokens, skip_special_tokens=True))

    entropy, conf_label, clusters = compute_semantic_entropy(candidates)
    return StabilityDiagnostics(
        lyapunov_exponent=lambda_exp,
        semantic_entropy=entropy,
        confidence_rating=conf_label,
        diagnostics_status="executed",
        cluster_count=len(set(clusters)) if clusters else 1
    )


def create_app(
    engine: Optional[LLMEngine] = None,
    tokenizer=None,
    model_name: str = "lyapunov-model"
) -> FastAPI:
    app = FastAPI(title="lyapunov-engine OpenAI-Compatible Server", version="0.1.0")

    # State
    app.state.engine = engine
    app.state.tokenizer = tokenizer
    app.state.model_name = model_name

    @app.get("/health")
    async def health():
        return {"status": "healthy", "engine": "lyapunov-engine"}

    @app.get("/v1/models")
    async def list_models():
        card = ModelCard(id=app.state.model_name)
        return ModelList(data=[card])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if app.state.engine is None or app.state.tokenizer is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Format chat prompt
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if hasattr(app.state.tokenizer, "apply_chat_template"):
            prompt_text = app.state.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "

        prompt_tokens = app.state.tokenizer.encode(prompt_text)

        sampling_params = SamplingParams(
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 1.0,
            top_k=req.top_k or -1,
            max_tokens=req.max_tokens or 128
        )

        request_id = str(uuid.uuid4())

        # Check for stability diagnostics
        stability_diag = None
        if req.include_stability_diagnostics:
            stability_diag = run_stability_pipeline(
                app.state.engine, app.state.tokenizer, prompt_tokens, max_tokens=req.max_tokens or 64
            )

        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                app.state.engine.add_request(request_id, prompt_tokens, sampling_params)
                prev_len = 0

                while app.state.engine.has_unfinished_requests():
                    outputs = app.state.engine.step()
                    for out in outputs:
                        if out.request_id == request_id:
                            curr_tokens = out.output_tokens
                            if len(curr_tokens) > prev_len:
                                new_tokens = curr_tokens[prev_len:]
                                text_chunk = app.state.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                prev_len = len(curr_tokens)

                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    model=app.state.model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionChunkDelta(content=text_chunk),
                                            finish_reason=out.finish_reason if out.is_finished else None
                                        )
                                    ],
                                    stability_diagnostics=stability_diag
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0.001)

                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming execution
        app.state.engine.add_request(request_id, prompt_tokens, sampling_params)
        final_output = None

        while app.state.engine.has_unfinished_requests():
            outputs = app.state.engine.step()
            for out in outputs:
                if out.request_id == request_id:
                    final_output = out
            await asyncio.sleep(0.001)

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")

        completion_text = app.state.tokenizer.decode(final_output.output_tokens, skip_special_tokens=True)

        return ChatCompletionResponse(
            id=request_id,
            model=app.state.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason=final_output.finish_reason or "stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(final_output.output_tokens),
                total_tokens=len(prompt_tokens) + len(final_output.output_tokens)
            ),
            stability_diagnostics=stability_diag
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if app.state.engine is None or app.state.tokenizer is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        if isinstance(req.prompt, str):
            prompt_tokens = app.state.tokenizer.encode(req.prompt)
        else:
            prompt_tokens = req.prompt

        sampling_params = SamplingParams(
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 1.0,
            top_k=req.top_k or -1,
            max_tokens=req.max_tokens or 128
        )

        request_id = str(uuid.uuid4())

        stability_diag = None
        if req.include_stability_diagnostics:
            stability_diag = run_stability_pipeline(
                app.state.engine, app.state.tokenizer, prompt_tokens, max_tokens=req.max_tokens or 64
            )

        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                app.state.engine.add_request(request_id, prompt_tokens, sampling_params)
                prev_len = 0

                while app.state.engine.has_unfinished_requests():
                    outputs = app.state.engine.step()
                    for out in outputs:
                        if out.request_id == request_id:
                            curr_tokens = out.output_tokens
                            if len(curr_tokens) > prev_len:
                                new_tokens = curr_tokens[prev_len:]
                                text_chunk = app.state.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                prev_len = len(curr_tokens)

                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    model=app.state.model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionChunkDelta(content=text_chunk),
                                            finish_reason=out.finish_reason if out.is_finished else None
                                        )
                                    ],
                                    stability_diagnostics=stability_diag
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0.001)

                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming execution
        app.state.engine.add_request(request_id, prompt_tokens, sampling_params)
        final_output = None

        while app.state.engine.has_unfinished_requests():
            outputs = app.state.engine.step()
            for out in outputs:
                if out.request_id == request_id:
                    final_output = out
            await asyncio.sleep(0.001)

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")

        completion_text = app.state.tokenizer.decode(final_output.output_tokens, skip_special_tokens=True)

        return CompletionResponse(
            id=request_id,
            model=app.state.model_name,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text=completion_text,
                    finish_reason=final_output.finish_reason or "stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(final_output.output_tokens),
                total_tokens=len(prompt_tokens) + len(final_output.output_tokens)
            ),
            stability_diagnostics=stability_diag
        )

    return app
