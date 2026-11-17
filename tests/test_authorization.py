from app.agent import get_order


def test_owner_sees_own_order():
    assert get_order(customer_id="C020", order_id="120") is not None


def test_other_customer_does_not_see_order():
    assert get_order(customer_id="C021", order_id="120") is None


def test_sql_injection_in_customer_id_returns_none():
    assert get_order(customer_id="C021' OR '1'='1", order_id="120") is None



