import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock
from sumospace.providers import ProviderRouter
from sumospace.exceptions import ProviderError

@pytest.mark.asyncio
async def test_provider_fallback_on_connect_error():
    router = ProviderRouter(provider="openai")
    
    # Mock primary to fail with ConnectError
    primary = AsyncMock()
    primary.complete.side_effect = httpx.ConnectError("Connection failed")
    primary.name = "openai"
    router._provider = primary
    
    # Mock secondary to succeed
    secondary = AsyncMock()
    secondary.complete.return_value = "Fallback success"
    secondary.name = "hf"
    router._secondary = secondary
    
    res = await router.complete(user="hi")
    assert res == "Fallback success"
    assert primary.complete.called
    assert secondary.complete.called

@pytest.mark.asyncio
async def test_provider_no_fallback_on_bad_request():
    router = ProviderRouter(provider="openai")
    
    # Mock primary to fail with 400 Bad Request
    response = MagicMock()
    response.status_code = 400
    err = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=response)
    
    primary = AsyncMock()
    primary.complete.side_effect = err
    router._provider = primary
    
    secondary = AsyncMock()
    router._secondary = secondary
    
    with pytest.raises(httpx.HTTPStatusError):
        await router.complete(user="hi")
    
    assert not secondary.complete.called

@pytest.mark.asyncio
async def test_provider_fallback_on_empty_response():
    router = ProviderRouter(provider="openai")
    
    # Mock primary to return empty string
    primary = AsyncMock()
    primary.complete.return_value = ""
    router._provider = primary
    
    # Mock secondary to succeed
    secondary = AsyncMock()
    secondary.complete.return_value = "Fallback success"
    router._secondary = secondary
    
    res = await router.complete(user="hi")
    assert res == "Fallback success"
