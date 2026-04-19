# How To Run the app.py Locally
[!NOTE] Must be in the local_deployment folder

## Using CPU:
Install PyTorch manually to specify version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

then run:

```pip install -r requirements.txt```

and finally, run:

```python app.py```

## Using GPU (only works on some machines):
Create virtual environment called ```mlenv``` using Python 3.11 (necessary for version of Torch to work):

```py -3.11 -m venv mlenv```

then activate using:

```mlenv\Scripts\activate```

Next, install GPU version of Torch (CUDA 12.1, used for GPU acceleration):

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

Now install requirements.txt:

```pip install -r requirements.txt```

and finally, run:

```python app.py```
