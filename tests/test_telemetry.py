import pytest
from unittest.mock import MagicMock, patch
from sumospace.telemetry import SumoTelemetry

def test_telemetry_disabled_is_noop():
    # Even if opentelemetry is missing, it shouldn't crash
    with patch("sumospace.telemetry._OTEL_AVAILABLE", False):
        tel = SumoTelemetry(enabled=True)
        assert tel.enabled is False
        
        with tel.span("test") as s:
            assert s is None

def test_telemetry_enabled_creates_spans():
    with patch("sumospace.telemetry._OTEL_AVAILABLE", True):
        mock_trace = MagicMock()
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        
        with patch("sumospace.telemetry.trace", mock_trace):
            with patch("opentelemetry.sdk.trace.TracerProvider"):
                with patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"):
                    with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"):
                        with patch("socket.create_connection"):
                            tel = SumoTelemetry(enabled=True, endpoint="http://localhost:4317")
                            assert tel.enabled is True
                            
                            with tel.span("my-span", attributes={"key": "val"}):
                                pass
                            
                            mock_tracer.start_as_current_span.assert_called_with("my-span")

@pytest.mark.asyncio
async def test_telemetry_async_span():
    with patch("sumospace.telemetry._OTEL_AVAILABLE", True):
        mock_trace = MagicMock()
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        
        with patch("sumospace.telemetry.trace", mock_trace):
            with patch("opentelemetry.sdk.trace.TracerProvider"):
                with patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"):
                    with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"):
                        with patch("socket.create_connection"):
                            tel = SumoTelemetry(enabled=True)
                            
                            async with tel.async_span("async-span"):
                                pass
                            
                            mock_tracer.start_as_current_span.assert_called_with("async-span")
