# `dose_response` 统计边界

本文件说明 `app/dose_response.py` 的估计对象、识别结论和必须声明的假设。它是实现说明，不替代 `tests/test_dose_response.py` 的可执行检查。

## 测量模型

对学习者 `i`、渠道 `c`、场合 `t`：

```
m[i,c,t] = T[i] + a[i] + u[i,c] + e[i,c,t]
```

- `T`：常规练习分钟（目标暴露）。
- `a`：同一学习者跨渠道共享的偏差（例如整体少报或代填者偏乐观）。
- `u`：渠道特有、跨场合持续偏差（例如设备安装方式固定偏差）。
- `e`：场合内暂态误差（每次填报的随机误差）。

对渠道均值取平均得到 `R`，记多渠道间协方差 `q = Cov(m_c1, m_c2)`、`S = mean(q)`、`V = Var(R)`：

```
q = Var(T) + Var(a) = S
V = Var(T) + Var(a) + Var(u) + Var(e)
```

**识别结论**：多渠道一致（重复填报、自填＋代填＋设备同向）只能识别 `S`，不能把 `Var(T)` 与 `Var(a)` 分开。重复次数增加只压低 `e` 的贡献，不改变这一结论。因此默认输出不假设“一致即真值”。

## 默认输出

1. `association.observed`：以练习前成绩线性调整后，后续成绩对观察投入的 OLS 关联。它受衰减影响，是关联而非收益。
2. `association.identified_range`：由 `S` 与观察方差得到的单侧识别界。方向随观察关联符号变化；另一侧保持无界（`null`）。`bound_estimate_interval_95` 只描述该界估计本身的自助抽样波动，不给出缺失那侧的区间。
3. `association.scenarios`：给定“共同偏差占 `S` 的比例”时校正后的关联。比例是显式假设，不是数据结论；`0` 仅是网格端点，不是默认答案。
4. `association.within_household`：同一家户两学习者的差分对比。它消掉家户共享因素，但仍含学习者层面的选择，因此也不是因果收益。
5. `association.audit_anchors`：把人工区间/设备计时当工具变量。锚点自身误差被抵消，因此得到的是“若锚点误差与真实分钟、成绩误差、共同偏差均独立”这一假设下的关联及隐含复合可靠性。核验不是随机抽人，使用声明的入选概率做逆概率加权；未加权与加权结果都公开，便于发现声明概率与实际选择不符。
6. `support.bins` / `supported_intervals`：观察投入的经验支持区间与相应成绩残差，按要求的最小人数标记。它们描述资料覆盖与描述性差异，不是校正曲线；稀疏或空箱不填充。

## 必须声明的假设

- 暂态误差与渠道特有偏差在渠道间相互独立；同一家户内的暂态误差可以相关，结果给出估计与区间（`diagnostics.household_transient_error`）。
- 共同偏差 `a` 本身不直接预测后续成绩，只通过真实练习 `T` 与成绩相关。若“偏乐观填报”同时预示更好成绩，则单侧界的方向也可能被破坏。`identified_range.assumptions` 明示此点。
- 工具变量锚点假设见上；该假设无法由资料自身验证。锚点结果与一致方差上界冲突时，`diagnostics.audit_consistency.reliability_check` 标出冲突，而不是选一边当校正。
- 所有结果均为关联。练习不是随机分配，未观测动机与选择仍在，`association.causal_status` 固定标注为非因果。

## 退化资料

`diagnostics.degenerate_conditions` 会列出：无投入变异、无多渠道一致信息、无核验、无重复测量、无家庭内对照、一致方差超过复合方差。对应量以 `null` 和 `status` 说明，而不是补零或填默认曲线。

## 可复现性

`uncertainty.seed`、`replicates_requested`、`replicates_valid` 与选项一起保存在结果 envelope 中；默认 1000 次家户级自助，固定种子，结果内容可直接以 `sha256` 幂等保存与回读。
