import torch
import matplotlib.pyplot as plt  # 맷플롯립사용

g_oldLogType = "none"  # none, single, multi


def to_str(value):
    valueTypeStr = "{}".format(type(value))
    resStr = ""
    is_torch_nn_modules = False

    try:
        getattr(value, "parameters")
        is_torch_nn_modules = True
    except:
        is_torch_nn_modules = False

    if is_torch_nn_modules:
        resStr += "{} \n".format(value)

        for key in value.state_dict():
            param = value.state_dict()[key]
            dataStr = "{}".format(param.data)
            if len(dataStr) > 100:
                dataStr = dataStr[:100] + " ..."
            resStr += "{}    {}\n    {}\n".format(key, param.shape, dataStr)

    elif "torch.Tensor" in valueTypeStr:
        dataStr = "{}".format(value.data)
        if len(dataStr) > 100:
            dataStr = dataStr[:100] + " ..."
        resStr += "{} {}\n".format(value.shape, dataStr)
    else:
        resStr += "{}".format(value)

    return resStr


def log(name, value):
    global g_oldLogType
    valueStr = to_str(value)

    if "\n" in valueStr:
        if g_oldLogType != "multi":
            print("")
        g_oldLogType = "multi"
        print("{} : ".format(name))

        for line in valueStr.splitlines():
            print("    {}".format(line))

        print("")
    else:
        g_oldLogType = "single"
        print("{} : {}".format(name, valueStr))


g_epoch_array = []
g_cost_array = []
g_accurcy_array = []


def log_epoch(epoch, nb_epoches, cost, accuracy=None):
    global g_epoch_array
    global g_cost_array
    global g_accurcy_array

    g_epoch_array.append(epoch)
    g_cost_array.append(cost.item())

    logStr = ""
    logStr += "{} \n".format("-" * 80)
    logStr += "epoch : {:4d}/{} \n".format(epoch, nb_epoches)
    logStr += "cost : {:.6f} \n".format(cost.item())

    if accuracy != None:
        g_accurcy_array.append(accuracy)
        logStr += "accuracy : {:2.2f} \n".format(accuracy)

    print(logStr)


def plt_init():
    global g_epoch_array
    global g_cost_array
    global g_accurcy_array
    g_epoch_array = []
    g_cost_array = []
    g_accurcy_array = []


def plt_show():
    plt.plot(g_epoch_array, g_cost_array, label="cost")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    if len(g_accurcy_array) > 0:
        plt.plot(g_epoch_array, g_accurcy_array, label="accurcy")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()


def plt_img_show(data, w=None, h=None):
    if (w is None) or (h is None):
        plt.imshow(data, cmap="Greys", interpolation="nearest")
    else:
        plt.imshow(data.view(w, h), cmap="Greys", interpolation="nearest")
    plt.show()


def get_regression_accuracy(hypothesis, y):
    y_not_0 = y.clone()
    y_not_0[y == 0.] = 1e-10

    diff = torch.abs(
        (hypothesis - y_not_0) / y_not_0
    )

    diff[diff > 1.] = 1.

    accuracy = torch.mean(
        torch.ones_like(hypothesis) -
        diff
    ).item()

    return accuracy


def get_binary_classification_accuracy(hypothesis, y):
    hypothesis_one_zero = (hypothesis > 0.5).float()
    accuracy = len(y[y == hypothesis_one_zero]) / len(y)
    return accuracy


def get_cross_entropy_accuracy(hypothesis_softmax, y):
    hypothesis_argmax_dim1 = torch.argmax(hypothesis_softmax, dim=1)
    accuracy = len(y[y == hypothesis_argmax_dim1]) / len(y)
    return accuracy
