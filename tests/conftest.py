import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Створюємо event loop для асинхронних тестів"""
    import asyncio

    policy = asyncio.WindowsSelectorEventLoopPolicy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
