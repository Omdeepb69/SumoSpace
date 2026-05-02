import time
import socket
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Any, Dict
from rich.console import Console

console = Console()

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

class SumoTelemetry:
    """
    Handles OpenTelemetry instrumentation.
    Provides context managers for spans that become no-ops if telemetry is disabled
    or the required packages are missing.
    """
    def __init__(
        self, 
        enabled: bool = False, 
        endpoint: str = "http://localhost:4317",
        service_name: str = "sumospace"
    ):
        self.enabled = enabled and _OTEL_AVAILABLE
        self._tracer = None
        
        if self.enabled:
            try:
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                
                # Check if collector is reachable (non-blocking in executor)
                if endpoint:
                    import concurrent.futures
                    def _check():
                        try:
                            host_port = endpoint.split("//")[-1].split(":")
                            host = host_port[0]
                            port = int(host_port[1]) if len(host_port) > 1 else 4317
                            with socket.create_connection((host, port), timeout=0.5):
                                return True
                        except Exception:
                            return False
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_check)
                        try:
                            reachable = future.result(timeout=1.0)
                            if not reachable:
                                console.print(
                                    f"[yellow]Telemetry warning: OTLP collector at {endpoint} "
                                    f"is unreachable. Spans will be queued and dropped.[/yellow]"
                                )
                        except Exception:
                            console.print(
                                f"[yellow]Telemetry warning: OTLP connectivity check timed out. "
                                f"Collector at {endpoint} may be unreachable.[/yellow]"
                            )

                resource = Resource.create({"service.name": service_name})
                provider = TracerProvider(resource=resource)
                
                exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
                
                trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer("sumospace")
                
            except ImportError:
                console.print(
                    "[yellow]Telemetry enabled but opentelemetry-sdk not installed. "
                    "Run: pip install sumospace[telemetry][/yellow]"
                )
                self.enabled = False
            except Exception as e:
                console.print(f"[yellow]Failed to initialize telemetry: {e}[/yellow]")
                self.enabled = False

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Synchronous span context manager."""
        if not self.enabled or not self._tracer:
            yield None
            return
            
        with self._tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, str(v))
            yield s

    @asynccontextmanager
    async def async_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Asynchronous span context manager."""
        if not self.enabled or not self._tracer:
            yield None
            return
            
        with self._tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, str(v))
            yield s
