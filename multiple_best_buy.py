
def maxProfit(prices: list[int], fee: int) -> int:
    vh, v = -prices[0], 0
    for p in prices[1:]:
        vh, v = max(vh, v - p), max(p + vh - fee, vh, v)
    return v

print(maxProfit(prices=[1, 3, 2, 8, 4, 9], fee=2))  # 8

