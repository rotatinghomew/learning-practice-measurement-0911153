from app.measurements import measurement_profile
from collections import defaultdict
from app.protocol import finite, table_mean, unique


def validate(packet):
    if not packet["revision"]:
        raise ValueError("missing revision")
    unique(packet["learners"], "learner")
    learners = {row["learner"] for row in packet["learners"]}
    for row in packet["observations"]:
        if row["learner"] not in learners or row["channel"] not in packet["channels"]:
            raise ValueError("unknown learner or channel")
        if row["minutes"] is not None and finite(row["minutes"], "minutes") < 0:
            raise ValueError("negative minutes")
    for row in packet["audits"]:
        if row["learner"] not in learners or not 0 < row["selection_probability"] <= 1:
            raise ValueError("invalid audit membership")


def repeatability(packet, options):
    groups = defaultdict(list)
    for row in packet["observations"]:
        groups[(row["learner"], row["channel"])].append(row["minutes"])
    channels = []
    for channel in packet["channels"]:
        differences = []
        for (learner, source), values in groups.items():
            observed = [value for value in values if value is not None]
            if source == channel and len(observed) >= 2:
                differences.append(abs(observed[0] - observed[1]))
        channels.append({"channel": channel, "repeat_pairs": len(differences),
                         "mean_absolute_repeat_difference": table_mean(differences)})
    return {"source_profile": measurement_profile(packet), "channels": channels, "audited_learners": len(packet["audits"]),
            "interpretation": "Within-channel repeatability; agreement does not establish measurement accuracy."}


ANALYSES = {"repeatability": repeatability}
