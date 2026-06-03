from logger import configure_logging
from config import Settings
from auth import TokenManager
from api_client import ApiClient
from trader import AutoTrader


def main() -> None:
    configure_logging()

    settings = Settings.load()
    token_manager = TokenManager(settings)
    api_client = ApiClient(settings, token_manager)

    trader = AutoTrader(settings, api_client)
    trader.run()


if __name__ == "__main__":
    main()
