# MLTools

For linux:

```
export MLTOOLS_PREFIX='/path/to/your/MLTools'
conda env create --name DS --file=$MLTOOLS_PREFIX/install/requirements.yml
conda activate DS
```

```
$MLTOOLS_PREFIX/install/conda_init.sh
```

For Windows
```
set "MLTOOLS_PREFIX=C:\path\to\your\\MLTools"
conda env create --name DS --file=%MLTOOLS_PREFIX%\install\requirements.yml
conda activate DS
```

```
%MLTOOLS_PREFIX%\install\conda_init.bat
```

```
pip install ipykernel
python -m ipykernel install --user --name=DS
```