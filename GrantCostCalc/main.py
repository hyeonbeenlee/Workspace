from payment_standard import *

def print_line():
    print(f"{'':=^50}")

def print_result():
    print_line()
    print(f"인건비: {round(cost_h):,}")
    for pos, fee in pay_pos.items():
        print(f"{pos:>10s}: {n_workers[pos]}인 * {pay_daily[pos]:,}원 * {period_days}일 * {pay_rates[pos] * 100:.2f}% = {round(pay_pos[pos]):,}원")
    print_line()
    print(f"직접경비: {round(cost_d):,}")
    print_line()
    print(f"간접비: {round(cost_i):,}")
    print_line()
    print(f"부가가치세: {round(cost_v):,}")
    print_line()
    print(f"총계: {round(re_cost_total):,}")
    print_line()

# Input data here
period_month = 8  # 과제수행기간(월)
cost_total = 2750  # 과제총액(만원)
cost_d = 200  # 직접비(만원)
n_workers = {'책임연구원': 1,
             '연구원': 0,
             '연구보조원': 1,
             '보조원': 0}

# Computing codes
period_days = period_month * 22
cost_total *= 10000
cost_d *= 10000
cost_hd = cost_total / 1.166  # 인건비+직접비
cost_h = cost_hd - cost_d  # 인건비
cost_i = cost_hd * 0.06  # 간접비
cost_v = (cost_hd + cost_i) * 0.1  # 부가세
re_cost_total = round(cost_h + cost_d + cost_i + cost_v)  # 전체 총계
assert re_cost_total == cost_total
# Computing codes
n_workers_total = sum(n_workers.values())  # 총 연구자수
pay_rates = {pos: cost_h / (n_workers_total * daily_fee * period_days) for pos, daily_fee in
             pay_daily.items()}  # 각 직급 적용비율
pay_pos = {pos: n_workers[pos] * pay_daily[pos] * period_days * rate for pos, rate in
           pay_rates.items()}  # 각 직급
re_cost_h = sum(pay_pos.values())  # 총인건비
assert re_cost_h == cost_h

print_result()

