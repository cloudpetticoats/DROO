import numpy as np


def sum_rate_single_WD(h, m, x, weights=None):
    """
    :param h: channel gain
    :param m: offloading decision
    :param x: a = x[0], and tau_j = a[1:]
    :param weights: WDs weight
    :return:
    """
    # 数据处理
    h = h / 1000000
    x_copy = [x[0]]
    for _ in range(len(m)):
        if m[_] == 1:
            if x[_+1] == 0.0:
                x_copy.append(1e-10)
            else:
                x_copy.append(x[_+1])
    x = x_copy
    del x_copy

    o = 100  # 处理一位任务数据所需的周期数
    p = 3  # AP发射功率
    u = 0.7  # 能量收集效率
    eta1 = ((u * p) ** (1.0 / 3)) / o
    ki = 10 ** -26  # 计算能效系数
    eta2 = u * p / 10 ** -10
    B = 2 * 10 ** 6  # 通信带宽
    Vu = 1.1  # 通信额外开销
    epsilon = B / (Vu * np.log(2))

    M0 = np.where(m == 0)[0]
    M1 = np.where(m == 1)[0]

    hi = np.array([h[i] for i in M0])
    hj = np.array([h[i] for i in M1])

    if weights is None:
        # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]
        weights = [1.5 if i % 2 == 1 else 1 for i in range(len(m))]

    wi = np.array([weights[M0[i]] for i in range(len(M0))])
    wj = np.array([weights[M1[i]] for i in range(len(M1))])

    local_rate_list = wi * eta1 * (hi / ki) ** (1.0 / 3) * x[0] ** (1.0 / 3)
    edge_rate_list = []

    sum1 = sum(wi * eta1 * (hi / ki) ** (1.0 / 3) * x[0] ** (1.0 / 3))
    sum2 = 0
    for i in range(len(M1)):
        rate_temp = wj[i] * epsilon * x[i + 1] * np.log(1 + eta2 * hj[i] ** 2 * x[0] / x[i + 1])
        edge_rate_list.append(rate_temp)
        sum2 += rate_temp
    return sum1 + sum2, local_rate_list, edge_rate_list


def single_timeframe_amount_data(local_rate_list, edge_rate_list, m_list, remaining_time):
    amount_data = []
    p, q = 0, 0
    for i in m_list:
        if i == 0:
            amount_data.append(local_rate_list[p]*remaining_time)
            p += 1
        else:
            amount_data.append(edge_rate_list[q]*remaining_time)
            q += 1
    print('--------')
    print(m_list)
    print('--------')
    print(amount_data)
    print('--------')


def combine_allocation(x, a, m):
    result = [x]
    p = 0
    for i in m:
        if i == 0:
            result.append(0)
        else:
            result.append(a[p])
            p += 1
    return result