import pytest
from hello_langgraph import route_decision, Category, Node


@pytest.mark.parametrize("category, expected_node", [
        (Category.BIOMED.value, Node.DOCUMENT_PATH.value),
        (Category.CALCULATION.value, Node.CALCULATION_PATH.value),
        (Category.GENERAL.value, Node.GENERAL_PATH.value)
])
def test_route_decision_routes_to_correct_node(category, expected_node):
    state = {"classification": category}
    assert route_decision(state) == expected_node

def test_route_decision_raises_on_unrecognized_classification():
    state = {"classification": "NOT_A_REAL_CATEGORY"}
    with pytest.raises(ValueError):
        route_decision(state)