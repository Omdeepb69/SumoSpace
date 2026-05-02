"""
SumoSpace Advanced Telemetry
============================
SumoSpace has built-in OpenTelemetry (OTLP) support.
This example shows how to enable it and view the span hierarchy.
"""

import asyncio
from sumospace import SumoKernel, SumoSettings

async def main():
    # To use telemetry, ensure you have opentelemetry-exporter-otlp installed:
    # pip install opentelemetry-exporter-otlp

    settings = SumoSettings(
        provider="hf",
        # 1. Enable telemetry
        telemetry_enabled=True,
        # 2. Point it to your Jaeger / Zipkin / OpenTelemetry Collector
        telemetry_endpoint="http://localhost:4317",
    )

    # The kernel will automatically capture spans for:
    # - Classify (intent detection)
    # - RAG (retrieval)
    # - Committee (planner, critic, resolver)
    # - Execute (each tool run)
    
    async with SumoKernel(settings=settings) as kernel:
        print("Sending task to kernel (telemetry enabled)...")
        await kernel.run("Check what OS we are running on.")
        
    print("\nCheck your OpenTelemetry collector (e.g. Jaeger) at http://localhost:16686 to see the trace.")

if __name__ == "__main__":
    asyncio.run(main())
