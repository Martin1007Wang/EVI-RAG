from __future__ import annotations

from src.archive.policy import PolicyEncoder, PreparedPolicyContext
from src.archive.policy.action import gather_actions_from_csr_lock_free


def test_archive_policy_namespace_exports_canonical_symbols() -> None:
    assert PolicyEncoder.__name__ == "PolicyEncoder"
    assert PreparedPolicyContext.__name__ == "PreparedPolicyContext"
    assert (
        gather_actions_from_csr_lock_free.__name__
        == "gather_actions_from_csr_lock_free"
    )
