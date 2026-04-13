from src.execution.order_manager import OrderManager


def test_normalize_usdc_balance_decimal_string():
    payload = {"balance": "347.15321"}
    result = OrderManager._normalize_usdc_balance(payload)
    assert result == 347.15321


def test_normalize_usdc_balance_micro_units_heuristic():
    payload = {"balance": "347153210"}
    result = OrderManager._normalize_usdc_balance(payload)
    assert result == 347.15321


def test_normalize_usdc_balance_with_decimals_field():
    payload = {"balance": "12345678", "decimals": 6}
    result = OrderManager._normalize_usdc_balance(payload)
    assert result == 12.345678
