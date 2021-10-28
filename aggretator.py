def aggregate_grads(grads, backend): #聚合
    """Aggregate model gradients to models.

    Args:
        data: a list of grads' information 梯度的信息格式
            item format: 项的格式
                {
                    'n_samples': xxx,
                    'named_grads': xxx,
                }
    Return:
        named grads: {
            'layer_name1': grads1,
            'layer_name2': grads2,
            ...
        }
    """
    total_grads = {} #字典
    n_total_samples = 0 #
    for gradinfo in grads: #梯度数组
        n_samples = gradinfo['n_samples'] #查梯度字典
        for k, v in gradinfo['named_grads'].items(): #
            if k not in total_grads:
                total_grads[k] = [] #字典没有就新建

            total_grads[k].append(v * n_samples) #键值*
        n_total_samples += n_samples #

    gradients = {} #梯度字典
    for k, v in total_grads.items(): #键值对
        gradients[k] = backend.sum(v, dim=0) / n_total_samples #

    return gradients
