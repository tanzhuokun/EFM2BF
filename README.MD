## The model run is a four-step process

### Step 1: Install Dependencies
python 3.8.8
torch 1.11
nibabel 3.2.2
networkx 2.5.1
scikit-learn 0.24.1

### Step 2 Data preparation 
We download the data from the following connection before proceeding to the next step.
#### （1）Download data such as R-fMRI and mapping associations of brain connectome gene networks
- cole/ -- https://github.com/ColeLab/ColeAnticevicNetPartition 
- humanbrainmap/normalized_microarray_donor10021/ -- https://human.brain-map.org/api/v2/well_known_file_download/178238373 
- humanconnectome/ -- https://db.humanconnectome.org/app/action/ChooseDownloadResources?project=HCP_Resources&resource=GroupAvg&filePath=HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500_Eigenmaps.dtseries.zip 
#### （2）Download the PPI network from the string database
- 9606.protein.info.v11.0.txt and 9606.protein.links.detailed.v11.0.txt -- https://string-db.org/cgi/download?sessionId=%24input-%3E%7BsessionId%7D&species_text=Homo+sapiens 
#### （3）Download brain disease genetic data for positive and negative samples
- gene_disease_associations.tsv -- https://www.disgenet.org/downloads 
- HumanDo.obo -- https://github.com/DiseaseOntology/HumanDiseaseOntology/tree/main/src/ontology 

### Step 3 Multi-scale extraction
This part is mainly divided into two parts: RWR algorithm and enhanced adaptive daul-GCN extraction
#### (1) RWR algorithm extraction
The RWR algorithm is used to extract features from the processed data
#### （2）enhanced adaptive daul-GCN extraction
The data processed by the RWR algorithm will be used as input to run the main_1.py file in the muti-scale folder

### Step 4 Multi-network convergence
This part is divided into two parts, the first is to input multi-scale features, and then fuse the features of different scales of multiple networks, and the second is to optimize the PSO model to the SVM model to complete the prediction.
#### （1）muti-network fusion
The multi-scale features of each network extracted in the second step are input into enhanced adaptive SSAE for processing and fusion.
#### (2) PSO optimizes SVM
The PSO algorithm is constructed to optimize the hyperparameters of SVM. Where (1) and (2) only need to run main_2.py.




