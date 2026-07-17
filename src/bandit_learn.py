import os
import pyarrow.parquet as pq
from bandit_policy import VwAdfXGBPolicy

def vw_learn_from_delta_parquet(
    policy: VwAdfXGBPolicy,
    delta_train_parquet: str,
    ) -> int:
    if (not os.path.exists(delta_train_parquet)) or os.path.getsize(delta_train_parquet) == 0:
        return 0

    df = pq.read_table(delta_train_parquet).to_pandas()
    if df.empty:
        return 0

    learned = 0

    for r in df.itertuples(index=False):
        cost = float(1 - r.reward)  # binary reward -> CB cost
        labeled = policy._add_label_to_adf(list(r.comment), r.chosen_index, cost=cost, prob=float(r.propensity))

        ex = policy.vw.parse(labeled)
        policy.vw.learn(ex)
        policy.vw.finish_example(ex)

        learned += 1

    return learned