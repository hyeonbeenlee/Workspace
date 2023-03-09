# 2023년, 22일 50% 참여
pay_standard = {'책임연구원': 3496704,
                '연구원': 2681226,
                '연구보조원': 1792309,
                '보조원': 1344277}
pay_monthly = {k: v * 2 for k, v in pay_standard.items()}
pay_daily = {k: round(v * 2 / 22) for k, v in pay_standard.items()}
