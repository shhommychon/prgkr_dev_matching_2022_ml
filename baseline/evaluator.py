import torch

import pandas as pd


def eval_dataset1(model, dataset1_test_dataloader, save_model_dir=".model", device="cuda", save_submission=True):
    """dataset1의 test 데이터를 사용한 일반화 성능 확인
    """
    model.eval()
    
    predicted = []
    with torch.no_grad():
        for idx, x in enumerate(dataset1_test_dataloader):
            x = x[0].to(device)
            output = model(x.float())

            _, preds = torch.max(output, 1)
            predicted.extend(preds.cpu().numpy())
        torch.cuda.empty_cache()
    
    pd_preds = pd.DataFrame(predicted, columns=['predicted value'])
    if save_submission:
        pd_preds.to_csv('submission.csv')
    return pd_preds