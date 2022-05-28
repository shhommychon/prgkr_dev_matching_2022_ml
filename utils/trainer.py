import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import os

from utils.custom_metrics import macro_f1


def train_dataset0(
        model, dataset0_train_dataloader, dataset0_test_dataloader, 
        loss_function, optimizer, scheduler,
        epochs=20,
        save_model_dir=".model", device="cuda"
    ):
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    
    model.to(device)
    
    train_dataloader = {
        "train": dataset0_train_dataloader,
        "test": dataset0_test_dataloader
    }
    
    epochs = epochs

    loss_ce = loss_function
    optimizer = optimizer
    scheduler = scheduler
    
    best_test_maf1 = 0.
    best_test_accr = 0.
    best_test_loss = 9999.

    model.init_weights()

    for epoch in range(epochs):
        for phase in ["train", "test"]:
            running_loss = 0.
            running_accr = 0.
            running_f1 = 0.
            running_len = 0

            if phase == "train":
                model.train()
            elif phase == "test":
                model.eval()

            print(f"epoch #{epoch:0>3} {phase} phase - ", end='')
            for idx, (samples, labels) in enumerate(train_dataloader[phase]):
                X, y_true = samples.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    logits = model.forward(X)
                    _, y_pred = torch.max(logits, 1)

                    probs = F.softmax(logits, dim=-1)
                    loss = loss_ce(probs, y_true)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_accr += torch.sum(y_pred == y_true)
                running_f1 += macro_f1(y_pred, y_true, num_classes=16) * X.size(0)
                running_len += X.size(0)

            # 한 epoch이 모두 종료되었을 때,
            epoch_loss = running_loss / running_len
            epoch_accr = running_accr / running_len
            epoch_maf1 = running_f1 / running_len

            if phase == "test":
                scheduler.step(epoch_maf1)

            print(f"\n\tloss : {epoch_loss:.4f}, accuracy : {epoch_accr:.4f}, macro_f1 : {epoch_maf1:.4f}")
            if phase == "test" and best_test_accr < epoch_accr:
                best_test_accr = epoch_accr
            if phase == "test" and best_test_maf1 < epoch_maf1:
                best_test_maf1 = epoch_maf1
                torch.save(model.state_dict(), os.path.join(save_model_dir, "best_model.pt"))
            if phase == "test" and best_test_loss > epoch_loss:
                best_test_loss = epoch_loss

            if phase == "test": print()

    print("finished training")
    print(f"best score - loss : {best_test_loss:.4f}, accuracy : {best_test_accr:.4f}, macro_f1 : {best_test_maf1:.4f}")

    
def train_dataset1(
        model, dataset1_train_dataloader, 
        loss_function, optimizer, scheduler,
        epochs=50,
        save_model_dir=".model", device="cuda"
    ):
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    
    model.to(device)
    
    train_dataloader = {
        "train": dataset1_train_dataloader,
    }
    phase = "train"
    
    epochs = epochs

    loss_ce = loss_function
    optimizer = optimizer
    scheduler = scheduler
    
    best_train_maf1 = 0.
    best_train_accr = 0.
    best_train_loss = 9999.

    model.init_weights(init_only_fc=True)

    for epoch in range(epochs):
        # for phase in ["train"]:
        running_loss = 0.
        running_accr = 0.
        running_f1 = 0.
        running_len = 0

        # if phase == "train":
        #     model.train()

        print(f"epoch #{epoch:0>3} {phase} phase - ", end='')
        for idx, (samples, labels) in enumerate(train_dataloader[phase]):
            X, y_true = samples.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                logits = model.forward(X)
                _, y_pred = torch.max(logits, 1)

                probs = F.softmax(logits, dim=-1)
                loss = loss_ce(probs, y_true)

                # if phase == "train":
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_accr += torch.sum(y_pred == y_true)
                running_f1 += macro_f1(y_pred, y_true, num_classes=16) * X.size(0)
                running_len += X.size(0)

        # 한 epoch이 모두 종료되었을 때,
        epoch_loss = running_loss / running_len
        epoch_accr = running_accr / running_len
        epoch_maf1 = running_f1 / running_len

        # if phase == "test":
        #     scheduler.step(epoch_maf1)

        print(f"\n\tloss : {epoch_loss:.4f}, accuracy : {epoch_accr:.4f}, macro_f1 : {epoch_maf1:.4f}")
        if phase == "train" and best_train_accr < epoch_accr:
            best_test_accr = epoch_accr
        if phase == "train" and best_train_maf1 < epoch_maf1:
            best_test_maf1 = epoch_maf1
            torch.save(model.state_dict(), os.path.join(save_model_dir, "best_model_transferred.pt"))
        if phase == "train" and best_train_loss > epoch_loss:
            best_test_loss = epoch_loss

        # if phase == "test": print()
        print()

    print("finished training")
