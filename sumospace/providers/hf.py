from typing import AsyncIterator
from .base import BaseProvider, ProviderCapabilities

HF_DEFAULT_MODELS = {
    "default":   "microsoft/Phi-3-mini-4k-instruct",
    "fast":      "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "capable":   "mistralai/Mistral-7B-Instruct-v0.3",
    "code":      "Qwen/Qwen2.5-Coder-3B-Instruct",
    "reasoning": "microsoft/Phi-3-medium-4k-instruct",
}

class HuggingFaceProvider(BaseProvider):
    name = "hf"

    def __init__(
        self,
        model: str = "default",
        load_in_4bit: bool = False,
        device: str = "auto",
    ):
        self.capabilities = ProviderCapabilities(
            structured_output=False,
            tool_calling=False,
            preferred_fallback="xml"
        )
        self.model_id = HF_DEFAULT_MODELS.get(model, model)
        self.load_in_4bit = load_in_4bit
        self.device = device
        self._pipe = None

    async def initialize(self):
        import asyncio
        from rich.console import Console
        console = Console()
        console.print(
            f"[dim]Loading [cyan]{self.model_id}[/cyan] locally "
            f"(first run downloads ~2-4GB, cached after)...[/dim]"
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        import torch
        from transformers import pipeline, BitsAndBytesConfig

        bnb_config = None
        if self.load_in_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                pass

        if self.device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_map = "mps"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
        else:
            device_map = self.device
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16

        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

    async def complete(
        self,
        user: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        import asyncio

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
                return_full_text=False,
            ),
        )

        generated = result[0].get("generated_text", "")
        if isinstance(generated, list):
            generated = generated[-1].get("content", "") if generated else ""
        return generated.strip()

    async def stream(
        self, user: str, system: str = "", temperature: float = 0.2
    ) -> AsyncIterator[str]:
        result = await self.complete(user, system, temperature)
        yield result
