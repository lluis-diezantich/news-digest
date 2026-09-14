"""The 429 policy: which rate limits are worth waiting out, and for how long.

Google answers a per-minute and a per-day limit with the same status code, so
everything here is about reading the body to tell them apart. Getting it wrong
in either direction is a real failure: treating a per-minute limit as terminal
publishes an unenriched digest, and treating a per-day limit as transient sleeps
until the workflow times out.
"""

import json

from conftest import PER_DAY_429, PER_MINUTE_429

from newsdigest import ratelimit


def plan(payload, **kwargs):
    kwargs.setdefault("attempt", 0)
    kwargs.setdefault("budget", 3)
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return ratelimit.plan(body, **kwargs)


class TestClassification:
    def test_per_minute_limit_is_retried(self):
        assert plan(PER_MINUTE_429).retry

    def test_per_day_limit_is_not_retried(self):
        decision = plan(PER_DAY_429)
        assert not decision.retry
        assert "daily" in decision.reason

    def test_per_day_wins_however_much_budget_is_left(self):
        assert not plan(PER_DAY_429, budget=99).retry

    def test_daily_prose_is_read_when_no_structured_violation_is_sent(self):
        body = {"error": {"message": "Quota exceeded: requests per day limit reached"}}
        assert not plan(body).retry

    def test_quota_failure_without_a_daily_violation_is_per_minute(self):
        """A QuotaFailure we understood and that did not say "day" is transient."""
        body = {"error": {"details": [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}],
        }]}}
        assert plan(body).retry


class TestDelay:
    def test_retry_info_is_honoured(self):
        # 31s from the response, plus up to 1s of jitter.
        assert 31.0 <= plan(PER_MINUTE_429).delay < 32.0

    def test_fallback_schedule_escalates_without_retry_info(self):
        body = {"error": {"message": "rate limit"}}
        delays = [plan(body, attempt=i).delay for i in range(3)]
        assert delays[0] < delays[1] < delays[2]

    def test_fallback_schedule_saturates_rather_than_growing(self):
        body = {"error": {"message": "rate limit"}}
        assert plan(body, attempt=99).delay <= ratelimit.MAX_DELAY + 1.0

    def test_an_absurd_server_delay_is_capped(self):
        body = {"error": {"details": [{
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "86400s",
        }]}}
        assert plan(body).delay <= ratelimit.MAX_DELAY + 1.0


class TestBudgets:
    def test_per_request_budget_is_respected(self):
        decision = plan(PER_MINUTE_429, attempt=2, budget=2)
        assert not decision.retry
        assert "after 2 waits" in decision.reason

    def test_run_wait_budget_stops_further_waiting(self):
        decision = plan(PER_MINUTE_429, spent=590.0, cap=600.0)
        assert not decision.retry
        assert "wait budget" in decision.reason

    def test_a_wait_that_fits_the_budget_is_allowed(self):
        assert plan(PER_MINUTE_429, spent=100.0, cap=600.0).retry


class TestMalformedBodies:
    """A bad 429 body must not crash the path whose job is surviving one."""

    def test_empty_body_is_retried_as_transient(self):
        assert plan("").retry

    def test_html_error_page_is_retried_as_transient(self):
        assert plan("<html>429 Too Many Requests</html>").retry

    def test_unexpected_json_shapes_do_not_raise(self):
        for body in ("null", "[]", '{"error": "a string"}', '{"error": {"details": {}}}',
                     '{"error": {"details": [null, 7]}}'):
            assert plan(body).retry

    def test_unparseable_retry_delay_falls_back_to_the_schedule(self):
        body = {"error": {"details": [{
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "soon",
        }]}}
        decision = plan(body)
        assert decision.retry and decision.delay > 0
