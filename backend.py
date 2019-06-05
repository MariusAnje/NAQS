import torch
import torch.nn.functional as F


def fit(model, optimizer, train_data=None, val_data=None, epochs=40,
        verbose=True):
    for epoch in range(epochs):
        if train_data is not None:
            loss, acc = epoch_fit(model, train_data, optimizer)
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs}, " +
                      f"Train Loss: {loss}, Train Acc: {acc:6.3%}", end='')
        if val_data is not None:
            with torch.no_grad():
                loss, acc = epoch_fit(model, val_data)
            if verbose:
                print(f" Val Loss: {loss}, Val Acc: {acc:6.3%}")


def epoch_fit(model, data, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    running_loss, running_correction, running_total = 0, 0, 0
    for input_batch, label_batch in data:
        loss, correction = batch_fit(
            model, input_batch, label_batch, optimizer)
        running_loss += loss
        running_correction += correction
        running_total += input_batch.size(0)
    return running_loss / running_total, running_correction / running_total


def batch_fit(model, input_batch, label_batch, optimizer=None):
    output_batch = model(input_batch)
    _, prediction = torch.max(output_batch, 1)
    # print(prediction.shape, label_batch.shape)
    loss = F.cross_entropy(output_batch, label_batch)
    correction = (prediction == label_batch).sum()
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), correction.item()


if __name__ == '__main__':
    import data
    import mnist_net
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model, optimizer = mnist_net.get_model(device)
    train_data, val_data = data.get_data('MNIST', device)
    fit(model, optimizer, train_data, val_data)
