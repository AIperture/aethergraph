import pytest 

slack_sdk = pytest.importorskip("slack_sdk")

def test_slack_sdk_imported():
    from aethergraph.plugins.channel.websockets.slack_ws import SlackSocketModeRunner
    assert SlackSocketModeRunner is not None