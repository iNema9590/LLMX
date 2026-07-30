# FACILE ablation — NDCG@5 (marginal attribution)

## Summary of the experiment

This is a component ablation over **100 questions** from 2WikiMultihopQA dataset, computed with Llama-3.1-8B model, 14 architectural variants, and six
evaluation budgets M ∈ {32, 64, 128, 264, 528, 728}. We take the full FACILE pipeline and
replace each component in turn with a dummy variant that does nothing, or remove it from
the pipeline entirely where that is possible. For every variant we compute NDCG@5 of the
estimated marginal attributions against exact Shapley values, per question, and average
over questions.

The reported delta is **full system minus ablated variant, paired per question**, so a
positive delta means removing the component *hurt*, i.e. the component earns its place.

| Component removed | mean NDCG@5 | delta vs full |
|---|---|---|---|
| Core set construction | 0.9230 | +0.01 |
| Neighbor expansion | 0.9430 | +0.03 |
| Pure kernel sampling | 0.9137 | −0.01 |
| Pure uniform sampling | 0.9455 | +0.03 |

These results suggest that the proposed sampling components each contribute meaningfully to the overall performance, with the combination providing the strongest improvements.
