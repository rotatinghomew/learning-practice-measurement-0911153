# 领域边界资料

这些小资料用于复核统计意义，不预定当前完整资料的效应方向或显著性。

- `data/cases/boundary-1.json`：各学习者投入相同且各渠道无变化，无法从变异估计投入响应。
- `data/cases/boundary-2.json`：取消核验资料，仍有重复填报；重复一致不证明不存在共同偏差。
- `data/cases/boundary-3.json`：各渠道共享按家庭变化的偏移，不应凭多渠道一致就宣布真值已确定。

## `dose_response` 对这些边界资料的行为

`python -m app.demo --analysis dose_response --packet <path>` 可逐个复核；下列为资料结构决定的输出，不是预设数值。

- `boundary-1`：投入无变异，观察关联不可估计；`support.supported_intervals` 为空，识别范围状态为 `not_identified`。核验资料不能补出曲线。
- `boundary-2`：重复填报仍可给出单侧识别范围（`partially_identified`，上界为 null），但所有核验锚点为 `null`，诊断标出 `no_audits`；共同偏差与真实变异的分配保持未定。
- `boundary-3`：多渠道德一致使 `agreement_reliability_upper_bound` 为 1，但核验锚点与之一致性检查会显示核验隐含的可靠性远低于该上界，并提示“一致不等于真值已确定”；家庭内无对照，`within_household` 为 `not_identified`。
