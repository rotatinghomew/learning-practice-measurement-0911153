from collections import defaultdict
from app.protocol import table_mean


def collapse_repeats(packet):
    grouped = defaultdict(list)
    for observation in packet["observations"]:
        grouped[(observation["learner"], observation["channel"])].append(observation)
    result = {}
    for key, observations in grouped.items():
        observed = [row["minutes"] for row in sorted(observations, key=lambda row: row["occasion"]) if row["minutes"] is not None]
        result[key] = {"mean": table_mean(observed), "observed": len(observed),
                       "missing": len(observations) - len(observed)}
    return result


def paired_agreement(first, second):
    differences = [left - right for left, right in zip(first, second)]
    average = table_mean(differences)
    variance = sum((value - average) ** 2 for value in differences) / (len(differences) - 1) if len(differences) > 1 else None
    return {"paired_learners": len(differences), "mean_signed_difference": average,
            "difference_variance": variance}


def measurement_profile(packet):
    collapsed = collapse_repeats(packet)
    channels = packet["channels"]
    agreement = []
    for first_index, first in enumerate(channels):
        for second in channels[first_index + 1:]:
            pairs = [(collapsed[(row["learner"], first)]["mean"], collapsed[(row["learner"], second)]["mean"])
                     for row in packet["learners"]]
            observed = [(left, right) for left, right in pairs if left is not None and right is not None]
            agreement.append({"channels": [first, second],
                              **paired_agreement([left for left, right in observed], [right for left, right in observed])})
    audited = {row["learner"] for row in packet["audits"]}
    households = defaultdict(list)
    for learner in packet["learners"]:
        households[learner["household"]].append(learner["learner"])
    return {"channel_agreement": agreement, "distinct_households": len(households),
            "audited_households": sum(bool(set(members) & audited) for members in households.values()),
            "audit_probability_range": [min(row["selection_probability"] for row in packet["audits"]),
                                        max(row["selection_probability"] for row in packet["audits"])] if audited else None,
            "reference_channel": None}
