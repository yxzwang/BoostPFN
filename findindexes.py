def max_indices(lista, listb):
    # 确保两个列表长度相同
    if len(lista) != len(listb):
        raise ValueError("两个列表必须具有相同的长度")

    # 将两个列表的索引按元素值的差值进行排序，差值大的排在前面
    differences = [(i, lista[i] - listb[i]) for i in range(len(lista))]
    differences.sort(key=lambda x: x[1], reverse=True)

    max_count = 0
    sum_lista = 0
    sum_listb = 0
    indices = []

    # 遍历排序后的差值列表，计算最多的index数目并记录索引
    for i, diff in differences:
        sum_lista += lista[i]
        sum_listb += listb[i]
        if sum_lista > sum_listb:
            max_count += 1
            indices.append(i)
        else:
            break
    indices.sort()
    return max_count, indices,differences

# 示例使用
lista = [0.9006,
0.8427,
0.8937,
0.7836,
0.7964,
0.756,
0.9536,
0.9378,
0.9538,
0.9924,
0.964,
0.9745,
0.5668,
0.8765,
0.8752,
0.7647,
0.8862,
0.8724,
1,
1,
1,
0.872,
1,
0.9811,
0.6574,
0.9606,
0.8768,
0.9907,
0.5811,
0.9824,
0.9606,
0.8775,
0.9972,
0.9938,]
listb = [0.9615,
0.8575,
0.916,
0.811,
0.8095,
0.8109,
0.9616,
0.989,
0.9625,
0.9935,
0.9701,
0.9827,
0.848,
0.8775,
0.877,
0.7924,
0.9138,
0.9018,
1,
1,
1,
0.8915,
0.9999,
0.9719,
0.6689,
0.9884,
0.8937,
0.9891,
0.6192,
0.9639,
0.8298,
0.8779,
0.9976,
0.9942,]
result = max_indices(lista, listb)
print("最多的index数目:", result[0])
print("这些index是:", result[1])
print("排序的index是",result[2])
