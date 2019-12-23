import utils as utils
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def main():
    args = utils.read_training_args()
    train_data = utils.get_transformed_data('flowers/train', 'train')
    test_data = utils.get_transformed_data('flowers/test', 'test')
    valdiation_data = utils.get_transformed_data('flowers/valid', 'validation')
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(valdiation_data, batch_size=50, shuffle=False)
    
    model = utils.get_pretrained_model(args.arch)
    model.classifier = utils.get_classifier(model.classifier.in_features, args.hidden_units)
    device = torch.device("cuda" if args.gpu == 'gpu' else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    trained_model = utils.train_model(train_loader, validation_loader, model, device, criterion, optimizer, args.epochs)
    utils.test_model(test_loader, trained_model, device)
    utils.save_checkpoint(args.save_dir, model, optimizer, args)
    
if __name__ == '__main__': main()
    
    
            
            