import utils as utils
import json as json
import torch

def main():
    args = utils.read_prediction_args()
    with open(args.category_names, 'r') as f: cat_to_name = json.load(f)
    model = utils.load_checkpoint(args.checkpoint)['model']
    device = torch.device('cuda' if args.gpu =='gpu' else 'cpu')
    input_image = utils.get_input_test_image(args.filepath)
    model = model.to(device)
    input_image = input_image.to(device)
    input_image = input_image.unsqueeze_(0)
    input_image = input_image.float()
    
    top_probs, top_labels, top_flowers = utils.predict(input_image, model, device, cat_to_name, args.top_k)
    utils.print_probability(top_flowers, top_probs)
    
if __name__ == '__main__': main()