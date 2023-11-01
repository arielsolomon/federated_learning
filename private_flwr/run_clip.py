from tqdm import tqdm
import argparse
from pathlib import Path
import clip
import torch
from torchvision.transforms import transforms
from dataloaders.data_loaders_factory import get_raw_data
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_command_line_arguments(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/datasets/cifar",
                        help="dir path for datafolder")

    parser.add_argument("--batch-size", type=int, default="512", help="Number of images in train batch")

    parser.add_argument("--use-cuda", type=bool, default=True,
                        help='Use GPU. Use cpu if not')

    args = parser.parse_args()
    return args

def save_fig(image, image_ood, total):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image)
    axarr[0].set_title('Native image')
    axarr[1].imshow(image_ood)
    axarr[1].set_title('ood_image')
    plt.savefig('Native_ood_' + str(total) + '.jpg')




def main():
    parser = argparse.ArgumentParser(description="Use CLIP for few shot")

    args = get_command_line_arguments(parser)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Load the model
    model, preprocess = clip.load('ViT-B/32', device)

    # Load data loaders for training and testing
    train_set, test_set, classes = get_raw_data(data_path=args.data_path)

    # Prepare the inputs
    # image, class_id = next(iter(train_loader))
    total = 0
    same_preds = 0
    correct = 0
    correct_ood = 0
    for image, label in tqdm(test_set):
        total += 1
        trans = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ColorJitter(contrast=0.5, brightness=1.0),
             transforms.ToPILImage()])
        #
        # trans = transforms.Compose(
        #         [transforms.ToTensor(),
        #          transforms.ColorJitter(contrast=0.1, brightness=1.0),
        #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #          transforms.ToPILImage()])
        image_ood = trans(image)
        # save_fig(image, image_ood, total)
        # if total==9:
        #     break

        image_input = preprocess(image).unsqueeze(0).to(device)
        image_input_ood = preprocess(image_ood).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features_ood = model.encode_image(image_input_ood)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_ood /= image_features_ood.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity_ood = (100.0 * image_features_ood @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        values_ood, indices_ood = similarity_ood[0].topk(1)
        same_preds += (indices == indices_ood).sum().item()
        correct += (indices == label).sum().item()
        correct_ood += (indices_ood == label).sum().item()
        # print('correct', correct,'out of', total)
        # print('correct ood', correct_ood,'out of', total)
        # print('same preds', same_preds,'out of', total)

    accuracy = 100 * correct / total
    accuracy_ood = 100 * correct_ood / total
    same_preds_ratio = 100 * same_preds / total

    print('\nNative accuracy: ',"%.1f" % accuracy)
    print('\nood accuracy: ',"%.1f" % accuracy_ood)
    print('\nsame_preds_ratio',"%.1f" % same_preds_ratio)


# Entry point
if __name__ == "__main__":
    main()
