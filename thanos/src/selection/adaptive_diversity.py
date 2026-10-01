"""Content-led selection adapted from Samet's SceneMind selector.

There are no mandatory act quotas: time windows and event groups only penalize
concentration when it actually occurs in the proposed selection.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def _knapsack(values: list[float], costs: list[int], budget: int) -> list[int]:
    dp = [0.0] * (budget + 1)
    take = [bytearray(budget + 1) for _ in values]
    for index, (value, cost) in enumerate(zip(values, costs)):
        if cost > budget:
            continue
        for capacity in range(budget, cost - 1, -1):
            candidate = dp[capacity - cost] + value
            if candidate > dp[capacity] + 1e-8:
                dp[capacity] = candidate
                take[index][capacity] = 1
    cursor = max(range(budget + 1), key=dp.__getitem__)
    selected = []
    for index in range(len(values) - 1, -1, -1):
        if cursor >= costs[index] and take[index][cursor]:
            selected.append(index)
            cursor -= costs[index]
    return selected[::-1]


def select_adaptive(
    rows: Iterable[Any],
    *,
    budget_sec: float,
    score: Callable[[Any], float],
    duration: Callable[[Any], float],
    position: Callable[[Any], float],
    event_group: Callable[[Any], Any],
    required: Callable[[Any], bool] | None = None,
) -> tuple[list[Any], dict]:
    """Iterative knapsack with Samet's soft 10-window / event-repeat penalties."""
    items = list(rows)
    if not items:
        return [], {"strategy": "samet_adaptive", "selected": 0}
    if budget_sec <= 0 or budget_sec >= 9999:
        return items, {"strategy": "samet_adaptive_all", "selected": len(items)}

    # Half-second resolution balances budget precision and long-video memory.
    tick = 0.5
    budget = max(1, int(budget_sec / tick))
    costs = [max(1, int(round(max(0.05, duration(row)) / tick))) for row in items]
    values = [max(0.000001, float(score(row))) for row in items]
    mandatory = [i for i, row in enumerate(items) if required and required(row)]
    mandatory_cost = sum(costs[i] for i in mandatory)
    if mandatory_cost > budget:
        raise ValueError("Korunması gereken sahneler hedef süreye sığmıyor")
    candidates = [i for i in range(len(items)) if i not in mandatory]
    adjusted = values[:]
    selected = mandatory[:]
    for _ in range(5):
        local = _knapsack(
            [adjusted[i] for i in candidates],
            [costs[i] for i in candidates],
            budget - mandatory_cost,
        )
        selected = mandatory + [candidates[i] for i in local]
        if not selected:
            break
        changed = False
        windows: dict[int, list[int]] = {}
        events: dict[Any, list[int]] = {}
        for index in selected:
            ratio = min(0.999999, max(0.0, position(items[index])))
            windows.setdefault(int(ratio * 10), []).append(index)
            events.setdefault(event_group(items[index]), []).append(index)
        for groups, limit, decay in ((windows, 0.34, 0.86), (events, 0.28, 0.78)):
            for members in groups.values():
                if sum(costs[i] for i in members) <= max(1, budget * limit):
                    continue
                running = 0
                for rank, index in enumerate(sorted(members, key=lambda i: values[i] / costs[i], reverse=True)):
                    running += costs[index]
                    if index in mandatory or rank == 0 or running <= budget * limit:
                        continue
                    lowered = max(values[index] * 0.30, adjusted[index] * decay)
                    if lowered < adjusted[index] - 1e-8:
                        adjusted[index] = lowered
                        changed = True
        if not changed:
            break
    ordered = sorted(selected, key=lambda i: position(items[i]))
    chosen = [items[i] for i in ordered]
    return chosen, {
        "strategy": "samet_adaptive",
        "selected": len(chosen),
        "selected_duration": round(sum(duration(row) for row in chosen), 3),
        "time_window_soft_limit": 0.34,
        "event_group_soft_limit": 0.28,
        "mandatory_count": len(mandatory),
    }
