import torch
import helper_api as ccm
import torch.nn
import torch.optim

def eval_model(model,test_data):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    pred_label = []
    ori_label = []

    for data in test_data:
        cell,cl = data
        if torch.cuda.is_available():
            cell = cell.cuda()
            cl = cl.cuda()

        out = model(cell)
        _, pred = torch.max(out, 1)
        num_correct = (pred == cl).sum()
        eval_acc += num_correct.item()

        p = pred.cpu().detach().numpy()
        pred_label.extend(list(p))
        c = cl.cpu().detach().numpy()
        ori_label.extend(list(c))
    e_l =  eval_loss
    e_a = eval_acc
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        e_l,
        e_a
    ))

    _ = ccm.compute_confusion_matrix(pred_label, ori_label)
def eval_model_version2(model,test_data):
    model.eval()
    eval_acc = 0
    pred_label = []
    ori_label = []

    for data in test_data:
        cell,cl = data
        if torch.cuda.is_available():
            cell = cell.cuda()
            cl = cl.cuda()

        out = model(cell)
        _, pred = torch.max(out, 1)
        num_correct = (pred == cl).sum()
        eval_acc += num_correct.item()

        p = pred.cpu().detach().numpy()
        pred_label.extend(list(p))
        c = cl.cpu().detach().numpy()
        ori_label.extend(list(c))
    e_a = eval_acc
    print('Acc: {:.6f}'.format(e_a ))

    cm = ccm.compute_confusion_matrix_ver2(pred_label, ori_label)
    acc_list = []
    for i in range(len(cm)):
        acc = cm[i][i]/cm[i][-1]
        acc_list.append(acc)
    return acc_list