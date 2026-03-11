import torch
import os
import argparse
import csv
from tools.utils import set_seed, set_save_path, BalancedBatchSizeIterator, save, load_adj, accuracy, EarlyStopping
from tools.run_tools import train_one_epoch_classifier, evaluate_one_epoch_classifier, create_net
from tools.datasets import load_single_subject

# Use the implemented FusionGNNNet
from models.FusionGNNNet import FusionGNNNet

def checkpoint_name(args, index):
    exp_tag = args.exp_type
    if "test" in args.exp_type:
        exp_tag = args.exp_type.replace("test", "train")
    
    return (
        f"{args.id}_{exp_tag}_fusiongnn_"
        f"h{args.hidden_dim}_l{args.num_layers}_k{args.topk}_{index}.pth.tar"
    )

def single_train(args, model, train_X, train_y, val_X, val_y, iterator, index):
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs/4)
    
    # Label smoothing as planned
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    early_stopper = EarlyStopping(patience=500, max_epochs=args.epochs)
    val_acc = 0

    print(f"Start training Subject {args.id} (FusionGNNNet)")
    
    for epoch in range(0, args.epochs):
        if early_stopper.early_stop:
            print(f'Early stop in epoch {epoch}')
            break

        train_one_epoch_classifier(iterator, (train_X, train_y), model, args.device, optim, criterion)
        scheduler.step()
        
        avg_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), model, args.device, criterion)
        early_stopper(avg_acc)
        
        save_checkpoints = {
            'model_classifier': model.state_dict(),
            'epoch': epoch + 1,
            'acc': avg_acc
        }
        
        if avg_acc > val_acc:
            val_acc = avg_acc
            save(save_checkpoints, os.path.join(args.model_path, checkpoint_name(args, index)))
            
    print(f'Subject {args.id} Train Done, Best Val Acc: {val_acc:.4f}')
    return model

def single_test(args, model, index, test_X, test_y):
    pth_file = checkpoint_name(args, index)
    checkpoint_path = os.path.join(args.model_path, pth_file)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device), weights_only=True)
    model.load_state_dict(checkpoint['model_classifier'])
    model.eval()
    model = model.to(args.device)
    
    with torch.no_grad():
        output, _ = model(test_X.to(args.device))
        test_acc, _ = accuracy(output.detach(), test_y.to(args.device).detach())
        test_acc = test_acc[0][0].item()
        
    print(f'Subject {args.id} | {pth_file} | Test Acc: {test_acc:.4f}')
    
    with open(f'{args.father_path}/{args.exp_type}_{args.duration}_{args.dataset}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.id, "fusiongnn", test_acc])

def run(args):
    # Setup
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)

    # Data Loading
    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(
        dataset=args.dataset, subject_id=args.id, to_tensor=True, duration=args.duration
    )
    
    # Split test set half into validation
    val_X = test_X[:len(test_X)//2]
    val_y = test_y[:len(test_y)//2]
    test_X = test_X[len(test_X)//2:]
    test_y = test_y[len(test_y)//2:]
    
    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)    
    Adj, centrality = load_adj(args.dataset)
    Adj = torch.tensor(Adj, dtype=torch.float32)
    centrality = torch.tensor(centrality, dtype=torch.int64)

    # Model Setup
    n_classes = len(torch.unique(train_y))
    model_kwargs = dict(
        Adj=Adj,
        eu_adj=eu_adj,
        centrality=centrality,
        in_chans=train_X.shape[1],
        n_classes=n_classes,
        input_time_length=train_X.shape[2],
        drop_prob=args.dropout,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        topk=args.topk,
        cnn_dim=args.cnn_dim,
    )
    
    models = create_net(FusionGNNNet, args.exp_type, **model_kwargs)
               
    # Training
    if "train" in args.exp_type:
        print(f"Type: {args.exp_type}, Subject: {args.id}, wd: {args.w_decay}, dropout: {args.dropout}")
        for i, model in enumerate(models):
            models[i] = single_train(args, model, train_X, train_y, val_X, val_y, iterator, i)

    # Testing
    for i, model in enumerate(models):
        single_test(args, model, i, test_X, test_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionGNNNet Training Script")
    parser.add_argument('-exp_type', type=str, default="train", choices=['train', 'test'])
    parser.add_argument('-dataset', type=str, default='bciciv2a', choices=['bciciv2a', 'bciciv2b'])
    parser.add_argument('-subject_id', type=int, default=1, help='Subject ID. Run all subjects if <= 0.')
    
    parser.add_argument('-epochs', type=int, default=1500, help='Number of epochs to train.')    
    parser.add_argument('-batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('-lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('-w_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('-label_smoothing', type=float, default=0.1, help='Label smoothing factor.')
    
    parser.add_argument('-hidden_dim', type=int, default=64, help='Hidden dimension size.')
    parser.add_argument('-num_layers', type=int, default=3, help='Number of GIN layers.')
    parser.add_argument('-topk', type=int, default=8, help='Top-k neighbors for dynamic graph.')
    parser.add_argument('-cnn_dim', type=int, default=64, help='CNN feature dimension per channel.')
    
    parser.add_argument('-duration', type=float, default=4.0, help='Duration of MI trials.')
    parser.add_argument('-seed', type=int, default=42, help='Random seed.')
    
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save path formulation
    args.father_path = f'./{args.dataset}_fusiongnn_{args.exp_type}_checkpoint'

    if args.dataset == 'bciciv2a':
        args.dataset = 'BNCI2014001'
        subject_ids = [args.subject_id] if args.subject_id > 0 else list(range(1, 10))
    elif args.dataset == 'bciciv2b':
        args.dataset = 'BNCI2014004'
        subject_ids = [args.subject_id] if args.subject_id > 0 else list(range(1, 10))
        
    for sid in subject_ids:
        args.id = sid
        run(args)
