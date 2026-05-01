import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_COTRAIN_SLOW") != "1",
    reason="Set RUN_COTRAIN_SLOW=1 to run the real-model smoke",
)


def test_real_student_generates_json_on_4_seed_fixture(tmp_path):
    """Smoke: run CoTrainEngine for 1 cycle on a 16-row Banking77 sliver.

    Requires GPU + cached HF model weights for Qwen2.5-3B-Instruct and
    Phi-3.5-mini-instruct. Asserts:
      - cotrain_ledger.jsonl exists and has > 0 rows
      - gate_rho.json computable from the run dir
      - audit ledger contains at least 1 peer_consensus or arbitration row
    """
    raise NotImplementedError("flesh out once Task 17 step 3 lands")
