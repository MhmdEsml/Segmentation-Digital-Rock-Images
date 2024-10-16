# Segmentation-Digital-Rock-Images
Welcome to the GitHub repository for semantic segmentation of 11 digital rock images from the Digital Rock Portal.

## Installing the Requirements and Running the Inference Script
**1. Setting Up Environment**

Before running the inference script, ensure you have the required dependencies installed. You can install them using the following command:

<div id="codeSnippet">
  <pre><code>pip install -r requirements.txt</code></pre>
  <button onclick="copyCode('pip install -r requirements.txt')"></button>
</div>

This command will install all the necessary Python packages listed in the requirements.txt file.

**2. Running Dataset_Download.py**

Once you have installed the dependencies you can download and prepare the dataset using the following command:

<div id="codeSnippet">
  <pre><code>python Dataset_Download.py</code></pre>
  <button onclick="copyCode('python Dataset_Download.py')"></button>
</div>

After running this script, you can choose the dataset, the numbers of training, validation, and test data.

These datasets are available:
- Berea
- Bandera Brown
- Bandera Gray
- Bentheimer
- Berea Sister Gray
- Berea Upper Gray
- Buff Berea
- CastleGate
- Kirby
- Leopard
- Parker

Example:
- Please enter the image you want to download (e.g., Berea): `Berea`
- Please enter the number of train patches: `1000`
- Please enter the number of validation patches: `100`
- Please enter the number of test patches: `100`

**3. Running train.py**

To train the residual UNET on the selected dataset, you can use the following command:

<div id="codeSnippet">
  <pre><code>python train.py</code></pre>
  <button onclick="copyCode('python train.py')"></button>
</div>

That's it! You have successfully trained a segmentation model on the selected data . Feel free to explore and analyze the output for your digital rock analysis needs.

**4. Results**

After finishing the training process, you can find the results in `./metrics` and `./predictions`
