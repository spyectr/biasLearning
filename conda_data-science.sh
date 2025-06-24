conda create -n data-science python=3.9
source activate data-science
pip install --upgrade pip
conda config --prepend channels apple
conda install -c apple tensorflow-deps
pip uninstall numpy
pip install numpy statsmodels
pip install tensorflow-macos
pip install ipykernel
pip install jupyterthemes
pip install tensorflow-metal
pip install tensorflow-datasets
pip install scikit-learn umap-learn
conda install matplotlib pandas seaborn 
conda install -c conda-forge jupyterlab pydot graphviz 
# pip install torch
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install -c conda-forge graph-tool
pip install --upgrade emnist

conda deactivate
echo 'Done with setup'