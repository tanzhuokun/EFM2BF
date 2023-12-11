import os 
import argparse
CUDA_LAUNCH_BLOCKING=1

id2disease = {
    "14330": "Parkinson disease",
    "1470": "Major depressive disorder",
}


def set_args():

    parser = argparse.ArgumentParser(description="Train the model to obtain features for The next process")
    parser.add_argument('--cfg_paths', type=str, default=['config_embedding_integration.json'],
                        nargs='+', help='The configs to train model.')
    parser.add_argument('--save_paths', type=str, default=[ '../output/integration'],
                        nargs='+', help="The path to save model and result. ")
    parser.add_argument('--disease_gene_paths', type=str, default=["all_data/temp_data/DOID_1470_genes.npy","all_data/temp_data/DOID_14330_genes.npy"],
                        nargs='+',  help=' ')
    args = parser.parse_args()

    return args


def obtain_gene_features(cfg_path, save_path):
    os.system('python  embedding.py'+' --cfg_path '+cfg_path+' --save_path '+save_path)

def evalution(save_path, disease_genes):
    files = os.listdir(save_path)
    files = [os.path.join(save_path, i) for i in files if i.endswith('.npz')]
    arg = " ".join(files)


    os.system('python PSO-SVM.py'+' --y_path '+disease_genes+' --x_path '+arg)

def main():
    args =  set_args()

    for cfg_path, save_path in zip(args.cfg_paths, args.save_paths):
        obtain_gene_features(cfg_path, save_path)

    for save_path in args.save_paths:

        for disease_gene_path in args.disease_gene_paths:
            disease_id = disease_gene_path.split('_')[1]
            print('### For predicting {} genes.'.format(id2disease[disease_id]))

            evalution(save_path, disease_gene_path)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print('########################################################')



if __name__ == "__main__":
    main()
